"""Build a Word review copy of the complete manuscript; final rendering is mandatory."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from PIL import Image


def plain(text):
    text = re.sub(r'\[([^]]+)\]\(([^)]+)\)', lambda m: m.group(2) if m.group(1) == 'Source' else m.group(1), text)
    return text.replace('**', '').replace('`', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(); section = doc.sections[0]
    section.page_width = Inches(8.5); section.page_height = Inches(11)
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2', 'Heading 3', 'Caption']:
        style = doc.styles[name]; style.font.name = 'Times New Roman'; style.font.size = Pt(12); style.font.color.rgb = RGBColor(0,0,0)
        style.paragraph_format.line_spacing = 2
        style.paragraph_format.space_after = Pt(0)
    for style in doc.styles:
        for border in style.element.xpath('./w:pPr/w:pBdr'):
            border.getparent().remove(border)
        for fonts in style.element.xpath('./w:rPr/w:rFonts'):
            for key in ['asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme']:
                fonts.attrib.pop(qn('w:'+key), None)
    doc.styles['Title'].font.bold = True
    footer = section.footer.paragraphs[0]; footer.alignment = 2
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    lines = args.source.read_text().splitlines(); i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('#'):
            level = len(line)-len(line.lstrip('#')); title = plain(line[level:].strip())
            doc.add_paragraph(title, style='Title' if level == 1 else 'Heading '+str(min(level-1,3)))
            i += 1; continue
        if line == r'\[':
            while lines[i].strip() != r'\]': i += 1
            ns = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
            def run(text): return '<m:r><m:t>'+escape(text)+'</m:t></m:r>'
            def sub(base, lower): return '<m:sSub><m:e>'+run(base)+'</m:e><m:sub>'+run(lower)+'</m:sub></m:sSub>'
            expression = sub('s','i,b')+run('(x) = ')
            expression += '<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="subSup"/><m:supHide m:val="1"/></m:naryPr><m:sub>'+run('t')+'</m:sub><m:sup/><m:e>'+run('log ')+sub('p','M')+run('(')+sub('c','it')+run(' | p, ')+sub('c','i,<t')+run(')')+'</m:e></m:nary>'
            p = doc.add_paragraph(); p._p.append(parse_xml('<m:oMath xmlns:m="'+ns+'">'+expression+'</m:oMath>'))
            i += 1; continue
        image = re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)', line)
        if image:
            path = args.source.parent/image.group(2)
            with Image.open(path) as im: width,height=im.size
            inches = min(6.5, 4.5*width/height)
            p=doc.add_paragraph();p.paragraph_format.line_spacing=1;p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(path),width=Inches(inches))
            caption=doc.add_paragraph(image.group(1),style='Caption');caption.paragraph_format.keep_with_next=False;i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[plain(c.strip()) for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
                i+=1
            table=doc.add_table(rows=0,cols=len(rows[0]));table.style='Table Grid'
            for index,row in enumerate(rows):
                if len(row)!=len(rows[0]):raise ValueError('Malformed manuscript table')
                cells=table.add_row().cells
                for cell,text in zip(cells,row):
                    cell.text=text
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing=2
                        for run in p.runs:run.font.size=Pt(10);run.bold=index==0
                if index==0:
                    repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![',r'\[')):
            paragraph.append(lines[i].strip());i+=1
        p = doc.add_paragraph()
        for token in re.split(r'(\*[^*]+\*)', plain(' '.join(paragraph))):
            run = p.add_run(token[1:-1] if token.startswith('*') and token.endswith('*') else token)
            run.italic = token.startswith('*') and token.endswith('*')
    doc.core_properties.title = doc.paragraphs[0].text
    doc.core_properties.author = ''
    doc.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
