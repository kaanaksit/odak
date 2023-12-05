import sys
import odak


def test():
    markdown_text = odak.tools.markdown(filename = './docs/installation.md')
    for line_id in range(markdown_text.get_line_count()):
        line, content_type = markdown_text.get_line(line_id = line_id)
        if content_type == 'text':
            print(line)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
