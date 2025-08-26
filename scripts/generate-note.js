const fs = require('fs');
const path = require('path');

const notesDir = path.join(__dirname, '..', 'notes');

if (!fs.existsSync(notesDir)) {
  fs.mkdirSync(notesDir, { recursive: true });
}

const files = fs.readdirSync(notesDir);

let maxDay = 0;
files.forEach((file) => {
  const match = file.match(/^day(\d{2})\.md$/);
  if (match) {
    const day = parseInt(match[1], 10);
    if (day > maxDay) {
      maxDay = day;
    }
  }
});

const nextDay = maxDay + 1;
const formattedDay = String(nextDay).padStart(2, '0');
const newFileName = `day${formattedDay}.md`;
const newFilePath = path.join(notesDir, newFileName);

const currentDate = new Date().toISOString().split('T')[0];

const template = `# 第 ${nextDay} 天

> ${currentDate}

## 学习的知识点

## 遇到的问题和解决方案

## 心得体会
`;

fs.writeFileSync(newFilePath, template);

console.log(`Successfully created note: ${newFilePath}`);
