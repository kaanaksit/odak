from .file import read_text_file


class latex():
    """
    A class to work with latex documents.
    """
    def __init__(
                 self,
                 filename
                ):
        """
        Parameters
        ----------
        filename     : str
                       Source filename (i.e. sample.tex).
        """
        self.filename = filename
        self.content = read_text_file(self.filename)
        self.content_type = []
        self.latex_dictionary = [
                                 '\\documentclass',
                                 '\\if',
                                 '\\pdf',
                                 '\\else',
                                 '\\fi',
                                 '\\vgtc',
                                 '\\teaser',
                                 '\\abstract',
                                 '\\CCS',
                                 '\\usepackage',
                                 '\\PassOptionsToPackage',
                                 '\\definecolor',
                                 '\\AtBeginDocument',
                                 '\\providecommand',
                                 '\\setcopyright',
                                 '\\copyrightyear',
                                 '\\acmYear',
                                 '\\citestyle',
                                 '\\newcommand',
                                 '\\acmDOI',
                                 '\\newabbreviation',
                                 '\\global',
                                 '\\begin{document}',
                                 '\\author',
                                 '\\affiliation',
                                 '\\email',
                                 '\\institution',
                                 '\\streetaddress',
                                 '\\city',
                                 '\\country',
                                 '\\postcode',
                                 '\\ccsdesc',
                                 '\\received',
                                 '\\includegraphics',
                                 '\\caption',
                                 '\\centering',
                                 '\\label',
                                 '\\maketitle',
                                 '\\toprule',
                                 '\\multirow',
                                 '\\multicolumn',
                                 '\\cmidrule',
                                 '\\addlinespace',
                                 '\\midrule',
                                 '\\cellcolor',
                                 '\\bibliography',
                                 '}',
                                 '\\title',
                                 '</ccs2012>',
                                 '\\bottomrule',
                                 '<concept>',
                                 '<concept',
                                 '<ccs',
                                 '\\item',
                                 '</concept',
                                 '\\begin{abstract}',
                                 '\\end{abstract}',
                                 '\\endinput',
                                 '\\\\'
                                ]
        self.latex_begin_dictionary = [
                                       '\\begin{figure}',
                                       '\\begin{figure*}',
                                       '\\begin{equation}',
                                       '\\begin{CCSXML}',
                                       '\\begin{teaserfigure}',
                                       '\\begin{table*}',
                                       '\\begin{table}',
                                       '\\begin{gather}',
                                       '\\begin{align}',
                                      ]
        self.latex_end_dictionary = [
                                     '\\end{figure}',
                                     '\\end{figure*}',
                                     '\\end{equation}',
                                     '\\end{CCSXML}',
                                     '\\end{teaserfigure}',
                                     '\\end{table*}',
                                     '\\end{table}',
                                     '\\end{gather}',
                                     '\\end{align}',
                                    ]
        self._label_lines()


    def set_latex_dictonaries(self, begin_dictionary, end_dictionary, syntax_dictionary):
        """
        Set document specific dictionaries so that the lines could be labelled in accordance.


        Parameters
        ----------
        begin_dictionary     : list
                               Pythonic list containing latex syntax for begin commands (i.e. \\begin{align}).
        end_dictionary       : list
                               Pythonic list containing latex syntax for end commands (i.e. \\end{table}).
        syntax_dictionary    : list
                               Pythonic list containing latex syntax (i.e. \\item).

        """
        self.latex_begin_dictionary = begin_dictionary
        self.latex_end_dictionary = end_dictionary
        self.latex_dictionary = syntax_dictionary
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
            for syntax in self.latex_begin_dictionary:
                if line.find(syntax) != -1:
                    content_type_flag = True
                    content_type = 'latex'
            for syntax in self.latex_dictionary:
                if line.find(syntax) != -1:
                    content_type = 'latex'
            if content_type_flag == True:
                content_type = 'latex'
                for syntax in self.latex_end_dictionary:
                    if line.find(syntax) != -1:
                         content_type_flag = False
            self.content_type.append(content_type)


    def get_line_count(self):
        """
        Definition to get the line count.


        Returns
        -------
        line_count     : int
                         Number of lines in the loaded latex document.
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
                         Line's content type (e.g., latex, comment, text).
        """
        line = self.content[line_id]
        content_type = self.content_type[line_id]
        return line, content_type
