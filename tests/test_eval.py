from eval_conversations import Turn, report_html, report_markdown, summarise


def test_eval_reports_render_without_an_llm():
    rows = [
        Turn(1, "Pong tvr ey", "Pg nham bay", 1.2, "primary", [], []),
        Turn(2, "Nham ey?", "Nham saj phat", 0.0, "primary", [], ["comma"]),
    ]
    results = {"food": {"turns": rows}}
    stats = summarise(results)
    assert stats["turns"] == 2
    assert stats["style_issues"] == 1
    assert stats["p95"] >= stats["p50"]

    markdown = report_markdown(results, stats)
    assert "## food" in markdown and "Pong tvr ey" in markdown

    page = report_html(results, stats)
    assert "Pong tvr ey" in page and "Copy review" in page
