from cs_agent.config import DATA_DIR
from cs_agent.rag import PolicyIndex


def index():
    return PolicyIndex.from_file(DATA_DIR / "policies.md")


def test_sections_are_split_by_heading():
    titles = [s.title for s in index().sections]
    assert "七天无理由退货" in titles and "人工客服" in titles
    assert all(s.body for s in index().sections)


def test_shipping_insurance_query():
    hits = index().search("退货运费谁出？有运费险吗", k=3)
    assert hits[0].title == "运费与运费险"


def test_custom_item_query():
    assert index().search("定制的东西能退吗")[0].title == "定制商品"


def test_warranty_query():
    assert index().search("保修期是多久")[0].title == "保修政策"


def test_price_protection_query():
    assert index().search("降价了能补差价吗")[0].title == "价格保护与补差价"


def test_empty_query_returns_nothing():
    assert index().search("   ") == []
    assert index().search("！？") == []
