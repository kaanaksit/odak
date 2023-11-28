from .file import read_text_file


class markdown():
    """
    A class to work with markdown documents.
    """
    def __init__(
                 self,
                 filename
                ):
        """
        Parameters
        ----------
        filename     : str
                       Source filename (i.e. sample.md).
        """
        self.filename = filename
        self.content = read_text_file(self.filename)
        self.content_type = []
        self.markdown_dictionary = [
                                     '#',
                                   ]
        self.markdown_begin_dictionary = [
                                          '```bash',
                                          '```python',
                                          '```',
                                         ]
        self.markdown_end_dictionary = [
                                        '```',
                                       ]
        self._label_lines()


    def set_dictonaries(self, begin_dictionary, end_dictionary, syntax_dictionary):
        """
        Set document specific dictionaries so that the lines could be labelled in accordance.


        Parameters
        ----------
        begin_dictionary     : list
                               Pythonic list containing markdown syntax for beginning of blocks (e.g., code, html).
        end_dictionary       : list
                               Pythonic list containing markdown syntax for end of blocks (e.g., code, html).
        syntax_dictionary    : list
                               Pythonic list containing markdown syntax (i.e. \\item).

        """
        self.markdown_begin_dictionary = begin_dictionary
        self.markdown_end_dictionary = end_dictionary
        self.markdown_dictionary = syntax_dictionary
        self._label_lines


    def _label_lines(self):
        """
        Internal function for labelling lines.
        """
        content_type_flag = False
        for line_id, line in enumerate(self.content):
            while len(line) > 0 and line[0] == ' ':
                 line = line[1::]
            self.content[line_id] = line
            if len(line) == 0:
                content_type = 'empty'
            elif line[0] == '%':
                content_type = 'comment'
            else:
                content_type = 'text'
            for syntax in self.markdown_begin_dictionary:
                if line.find(syntax) != -1:
                    content_type_flag = True
                    content_type = 'markdown'
            for syntax in self.markdown_dictionary:
                if line.find(syntax) != -1:
                    content_type = 'markdown'
            if content_type_flag == True:
                content_type = 'markdown'
                for syntax in self.markdown_end_dictionary:
                    if line.find(syntax) != -1:
                         content_type_flag = False
            self.content_type.append(content_type)


    def get_line_count(self):
        """
        Definition to get the line count.


        Returns
        -------
        line_count     : int
                         Number of lines in the loaded markdown document.
        """
        self.line_count = len(self.content)
        return self.line_count


    def get_line(self, line_id = 0):
        """
        Definition to get a specific line by inputting a line nunber.


        Returns
        ----------
        line           : str
                         Requested line.
        content_type   : str
                         Line's content type (e.g., markdown, comment, text).
        """
        line = self.content[line_id]
        content_type = self.content_type[line_id]
        return line, content_type
