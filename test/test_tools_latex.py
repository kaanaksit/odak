import sys
import odak


def test(
    header="test/test_tools_latex.py",
):
    manuscript = odak.tools.latex(filename="./test/data/sample_latex.tex")
    for line_id in range(manuscript.get_line_count()):
        line, content_type = manuscript.get_line(line_id=line_id)
        if content_type == "text":
            odak.log.logger.info("{} -> {}".format(header, line))
    assert True == True


if __name__ == "__main__":
    sys.exit(test())
