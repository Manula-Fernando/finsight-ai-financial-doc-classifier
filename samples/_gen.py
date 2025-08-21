from PIL import Image, ImageDraw, ImageFont

# Generate sample images for the current 5-class setup:
# bank_statement, salary_slip, income_tax_return, utility_bill, cheque
# (Purely synthetic placeholders; not used for training.)

_FONT = None  # optional future font handling

def _new_doc(w=900, h=1200, bg=(255, 255, 255)):
    return Image.new("RGB", (w, h), bg)


def make_bank_statement(path: str, idx: int):
    img = _new_doc()
    d = ImageDraw.Draw(img)
    d.text((50, 40), "BANK STATEMENT", fill=(0, 0, 0))
    d.text((50, 120), f"Account: 1234-{idx:04d}", fill=(0, 0, 0))
    d.text((50, 180), "Period: 2025-07-01 to 2025-07-31", fill=(0, 0, 0))
    d.text((50, 240), "Opening Balance: LKR 150,000.00", fill=(0, 0, 0))
    d.text((50, 300), "Closing Balance: LKR 163,456.78", fill=(0, 0, 0))
    d.text((50, 380), "Txn 01 2025-07-02  Grocery Store    -4,500.00", fill=(0,0,0))
    d.text((50, 420), "Txn 02 2025-07-03  Salary Credit   +25,000.00", fill=(0,0,0))
    img.save(path)


def make_salary_slip(path: str, idx: int):
    img = _new_doc()
    d = ImageDraw.Draw(img)
    d.text((50, 40), "SALARY SLIP", fill=(0, 0, 0))
    d.text((50, 120), f"Employee: EMP-{idx:05d}", fill=(0, 0, 0))
    d.text((50, 180), "Month: July 2025", fill=(0, 0, 0))
    d.text((50, 240), "Basic Pay: LKR 80,000.00", fill=(0, 0, 0))
    d.text((50, 300), "Allowances: LKR 20,000.00", fill=(0, 0, 0))
    d.text((50, 360), "Deductions: LKR 5,500.00", fill=(0, 0, 0))
    d.text((50, 420), "Net Pay: LKR 94,500.00", fill=(0, 0, 0))
    img.save(path)


def make_income_tax_return(path: str, idx: int):
    img = _new_doc()
    d = ImageDraw.Draw(img)
    d.text((50, 40), "INCOME TAX RETURN", fill=(0, 0, 0))
    d.text((50, 120), f"Filing ID: ITR-{idx:06d}", fill=(0, 0, 0))
    d.text((50, 180), "Assessment Year: 2024-2025", fill=(0, 0, 0))
    d.text((50, 240), "Total Income: LKR 1,200,000.00", fill=(0, 0, 0))
    d.text((50, 300), "Tax Paid: LKR 180,000.00", fill=(0, 0, 0))
    d.text((50, 360), "Refund Due: LKR 5,000.00", fill=(0, 0, 0))
    img.save(path)


def make_utility_bill(path: str, idx: int):
    img = _new_doc()
    d = ImageDraw.Draw(img)
    d.text((50, 40), "UTILITY BILL", fill=(0, 0, 0))
    d.text((50, 120), f"Account: UTIL-{idx:05d}", fill=(0, 0, 0))
    d.text((50, 180), "Service: Electricity", fill=(0, 0, 0))
    d.text((50, 240), "Billing Period: Jul 2025", fill=(0, 0, 0))
    d.text((50, 300), "Units: 350", fill=(0, 0, 0))
    d.text((50, 360), "Amount Due: Rs. 7,850.00", fill=(0, 0, 0))
    d.text((50, 420), "Due Date: 2025-08-15", fill=(0, 0, 0))
    img.save(path)

def make_cheque(path: str, idx: int):
    img = Image.new("RGB", (1200, 600), (245, 248, 250))
    d = ImageDraw.Draw(img)
    d.text((60, 40), "CHEQUE", fill=(0, 0, 0))
    d.text((60, 120), f"Cheque No: CHQ-{idx:06d}", fill=(0, 0, 0))
    d.text((60, 200), "Payee: JOHN DOE", fill=(0, 0, 0))
    d.text((60, 280), "Amount: Rs. 9,876.54", fill=(0, 0, 0))
    d.text((60, 360), "Date: 21/08/2025", fill=(0, 0, 0))
    img.save(path)


if __name__ == "__main__":
    import os
    root = os.path.dirname(__file__)
    os.makedirs(root, exist_ok=True)
    make_bank_statement(os.path.join(root, 'bank_statement_demo.jpg'), 1)
    make_salary_slip(os.path.join(root, 'salary_slip_demo.jpg'), 1)
    make_income_tax_return(os.path.join(root, 'income_tax_return_demo.jpg'), 1)
    make_utility_bill(os.path.join(root, 'utility_bill_demo.jpg'), 1)
    make_cheque(os.path.join(root, 'cheque_demo.jpg'), 1)
    print("Generated sample images for 5 classes in samples/")
