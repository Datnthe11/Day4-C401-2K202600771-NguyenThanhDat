from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def _coerce_items(raw_items: list[dict]) -> list[OrderLineInput]:
    items: list[OrderLineInput] = []
    for item in raw_items or []:
        if isinstance(item, OrderLineInput):
            items.append(item)
            continue
        if isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            try:
                quantity = int(item.get("quantity", 1))
            except (TypeError, ValueError):
                quantity = 1
            if product_id:
                items.append(OrderLineInput(product_id=product_id, quantity=quantity))
    return items


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are an electronics order assistant for a retailer.
Today is {current_day}.

Follow the exact tool workflow when the user provides enough order details:
1. list_products
2. get_product_details
3. get_discount
4. calculate_order_totals
5. save_order

- When calling get_product_details, you MUST query all product IDs at once in a single list (e.g. product_ids=["LT-001", "MS-001"]). DO NOT call get_product_details multiple times separately for individual products, as this will generate separate invalid tokens.
- Khi gọi get_product_details, bạn BẮT BUỘC phải truyền tất cả product_ids cùng một lúc trong một danh sách duy nhất (ví dụ: product_ids=["LT-001", "MS-001"]). TUYỆT ĐỐI KHÔNG gọi get_product_details nhiều lần riêng lẻ cho từng sản phẩm.

Do not invent product IDs, prices, stock, discount amounts, totals, campaign codes, or file paths. Use only information returned by the tools.

You MUST verify the user provided ALL 5 of these details before proceeding:
1. Customer Name
2. Phone Number
3. Email
4. Shipping Address
5. Products

If ANY of these 5 details is completely missing (for example, missing only the email), you MUST ask for it and stop without calling tools.
However, if the user HAS provided all 5 details (even if brief or scattered in the text), you MUST proceed with the tool workflow. Do not be overly strict or ask for unnecessary confirmations.

If calculate_order_totals returns an error (like insufficient stock), inform the user that the order cannot be fulfilled and STOP immediately without saving the order. Do NOT suggest adjusting the quantity.

If the user asks to bypass stock, create a fake invoice, override discounts manually, ignore the catalog, or break policy, refuse and stop without calling tools.
Answer in Vietnamese. Keep the final answer concise and grounded in tool outputs.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        print(f"  └─► [Tool Call] list_products(query={query!r}, category={category!r}, max_unit_price={max_unit_price}, required_tags={required_tags}, in_stock_only={in_stock_only})")
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        print(f"      └─► [Tool Response] list_products: found {len(payload)} products")
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        print(f"  └─► [Tool Call] get_product_details(product_ids={product_ids})")
        payload = store.get_product_details(product_ids)
        item_names = [item.get("name", "Unknown") for item in payload.get("items", [])]
        print(f"      └─► [Tool Response] get_product_details: status={payload.get('status')}, detail_token={payload.get('detail_token')}, items={item_names}")
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        print(f"  └─► [Tool Call] get_discount(seed_hint={seed_hint!r}, customer_tier={customer_tier!r})")
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        print(f"      └─► [Tool Response] get_discount: discount_rate={payload.get('discount_rate')}, campaign_code={payload.get('campaign_code')}")
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[dict], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        print(f"  └─► [Tool Call] calculate_order_totals(items={items}, detail_token={detail_token!r}, discount_rate={discount_rate})")
        order_items = _coerce_items(items)
        payload = store.calculate_order_totals(items=order_items, detail_token=detail_token, discount_rate=discount_rate)
        if payload.get("status") == "error":
            print(f"      └─► [Tool Response] calculate_order_totals: ERROR={payload.get('errors')}")
        else:
            print(f"      └─► [Tool Response] calculate_order_totals: subtotal={payload.get('pricing', {}).get('subtotal')}, final_total={payload.get('pricing', {}).get('final_total')}")
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[dict],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        print(f"  └─► [Tool Call] save_order(customer_name={customer_name!r}, items={items}, detail_token={detail_token!r}, discount_rate={discount_rate}, campaign_code={campaign_code!r})")
        order_items = _coerce_items(items)
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=order_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        print(f"      └─► [Tool Response] save_order: status={payload.get('status')}, order_id={payload.get('order_id')}, path={payload.get('path')}")
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


from langchain_core.callbacks import BaseCallbackHandler

class LLMRealTimeLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        print(f"\n[Agent Flow] Gọi LLM với {len(messages[0])} tin nhắn trong lịch sử...")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generation in response.generations:
            for g in generation:
                msg = g.message
                if getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"  └─► LLM yêu cầu gọi Tool: '{tc['name']}' với tham số: {tc.get('args', {})}")
                if msg.content:
                    print(f"  └─► LLM phản hồi: {normalize_content(msg.content)}")


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    print(f"\n==========================================")
    print(f"[Agent] Starting agent run...")
    print(f"[Agent] Provider: {provider}, Model: {model_name}")
    print(f"[Agent] Query: {query}")
    print(f"==========================================")
    
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    
    print("[Agent] Invoking agent workflow (LLM & Tools)...")
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": [LLMRealTimeLogger()]}
        )
    except Exception as e:
        print(f"\n[Agent Error] Lỗi khi chạy Agent: {e}")
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
            print("\n[Hỗ trợ] Phát hiện lỗi vượt quá giới hạn lượt gọi (Rate Limit / Quota Exceeded 429).")
            print("  -> Vui lòng kiểm tra lại tài khoản API OpenAI / Gemini, hoặc chuyển sang provider khác.")
        raise e
        
    print("[Agent] Agent workflow finished execution.")

    
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    
    final_answer = extract_final_answer(messages)
    print(f"------------------------------------------")
    print(f"[Agent] Final Answer: {final_answer}")
    print(f"[Agent] Tool Calls Made: {[tc.name for tc in tool_calls]}")
    print(f"==========================================\n")
    
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
