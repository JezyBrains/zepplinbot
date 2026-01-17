from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

class ReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Zeppelin Pro: Comprehensive Project Report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

def clean_text(text):
    """Replace common emojis and special characters with ASCII equivalents"""
    replacements = {
        '→': '->', '🟢': '[OK]', '🟡': '[WARN]', '🔴': '[CRIT]', 
        '🛑': '[STOP]', '⚠️': '[WARNING]', '💰': '[PROFIT]', 
        '🎰': '[GAME]', '🤖': '[AI]', '🏆': '[BEST]', '🚫': '[WORST]',
        '💥': '[CRASH]', '⚡': '[SPEED]', '👉': '>', '🕐': '[TIME]',
        '🌕': '[NIGHT]', '🔥': '[HOT]', '⏳': '[WAIT]', '👥': '[MASS]',
        '🎰': '[SLOT]', '📉': '[TREND]', '✅': '[YES]', '📊': '[DATA]',
        '∑': 'sum', '∆': 'delta', 'Δ': 'delta', 'π': 'pi',
        '…': '...', '–': '-', '—': '-', '“': '"', '”': '"', '‘': "'", '’': "'"
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode('ascii', 'ignore').decode('ascii')

def convert_md_to_pdf(md_path, pdf_path):
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('helvetica', '', 12)
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found")
        return

    with open(md_path, 'r') as f:
        for line in f:
            line = clean_text(line.strip())
            if not line:
                pdf.ln(5)
                continue
            
            if line.startswith('### '):
                pdf.set_font('helvetica', 'B', 12)
                pdf.cell(pdf.epw, 10, line[4:], border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('helvetica', '', 12)
            elif line.startswith('## '):
                pdf.ln(5)
                pdf.set_font('helvetica', 'B', 14)
                pdf.cell(pdf.epw, 10, line[3:], border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('helvetica', '', 12)
            elif line.startswith('# '):
                pass
            elif line.startswith('- '):
                pdf.set_x(20)
                pdf.multi_cell(pdf.epw - 10, 8, "* " + line[2:], border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            elif line.startswith('|'):
                pdf.set_font('courier', '', 9)
                pdf.multi_cell(pdf.epw, 8, line, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font('helvetica', '', 12)
            else:
                pdf.multi_cell(pdf.epw, 8, line, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(pdf_path)
    print(f"Successfully exported to {pdf_path}")

if __name__ == "__main__":
    src = "/Users/noobmaster69/.gemini/antigravity/brain/5ed245c0-3751-4099-b645-e0673de33de6/project_report.md"
    dest = "/Users/noobmaster69/.gemini/antigravity/brain/5ed245c0-3751-4099-b645-e0673de33de6/project_report.pdf"
    convert_md_to_pdf(src, dest)
