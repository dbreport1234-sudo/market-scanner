market-scanner
├── .github
│   └── workflows
│       └── scan.yml
├── lists                    -- โฟลเดอร์ใหม่สำหรับเก็บไฟล์รายชื่อแยก
│   ├── thai_stocks.json      -- รายชื่อหุ้นไทยทั้งหมด
│   ├── us_stocks.json        -- รายชื่อหุ้นสหรัฐฯ ทั้งหมด
│   └── crypto.json           -- รายชื่อคริปโตทั้งหมด
├── scanner.py                -- สคริปต์หลักที่จะอ่านค่าจากไฟล์เหล่านี้อัตโนมัติ
├── index.html
└── requirements.txt