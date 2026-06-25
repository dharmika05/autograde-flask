# 🎓 AutoGrade — Flask Automated Answer Grading System

Full-stack Flask web app with Teacher & Student portals, ML-powered grading,
6 built-in subjects, exam history, and progress tracking charts.

---

## 📁 Project Structure

```
autograde_flask/
├── app.py            ← All-in-one Flask app (models + routes + ML grader + seed)
├── requirements.txt
├── autograde.db      ← SQLite database (auto-created on first run)
└── templates/
    ├── base.html
    ├── auth/         ← home.html, login.html, register.html
    ├── teacher/      ← dashboard, add_question, create_exam, exam_results, student_detail
    └── student/      ← dashboard, take_exam, result, progress
```

---

## 🚀 Setup & Run (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app (auto-creates DB + seeds demo data)
python app.py

# 3. Open browser
# http://127.0.0.1:5000
```

---

## 🔐 Demo Login Credentials

| Role    | Username       | Password    |
|---------|---------------|-------------|
| Teacher | teacher_demo  | teacher123  |
| Student | student_asha  | asha123     |
| Student | student_rahul | rahul123    |
| Student | student_priya | priya123    |

---

## 📚 6 Subjects Pre-loaded (13 Questions)

| Subject              | Questions |
|----------------------|-----------|
| Computer Science     | 3         |
| Networking           | 2         |
| Database Management  | 2         |
| Mathematics          | 2         |
| Physics              | 2         |
| History              | 2         |

---

## 👩‍🏫 Teacher Features
- Dashboard with stats: questions, exams, students, subjects
- **Add Questions** to any subject (type new or pick existing)
- **Edit / Delete** questions
- **Create Exams** from your questions, toggle active/inactive
- **View Exam Results** — all submissions with grades and score bars
- **Student Detail** — per-question breakdown with ML scores and feedback

## 🎓 Student Features
- Dashboard — available exams + full exam history table
- **Take Exam** — 45-minute timer, live score preview as you type
- **Result Report** — animated grade ring, per-question score breakdown, AI feedback
- **Progress Page** — Chart.js bar chart + exam timeline with all past results

---

## 🤖 How ML Grading Works

1. **Preprocessing** — lowercase, remove stopwords, stem words
2. **TF-IDF Vectors** — weighted term-frequency for each answer
3. **Cosine Similarity (55%)** — semantic angle between student and reference
4. **Keyword Coverage (45%)** — checks domain-specific terms present
5. **Length Adequacy** — penalises very brief answers

| Grade | Percentage |
|-------|-----------|
| A+    | 90–100%   |
| A     | 80–89%    |
| B+    | 70–79%    |
| B     | 60–69%    |
| C     | 50–59%    |
| D     | 40–49%    |
| F     | 0–39%     |

---

## ➕ Adding More Questions

Log in as teacher → Dashboard → **Add Question** button.

Or add programmatically in `app.py` inside the `seed()` function's `SEED` list:

```python
("Your Subject Name", [
    ("Question text here?",
     "The complete model/reference answer with all key terms and concepts.",
     10),  # max marks
]),
```

Then delete `autograde.db` and re-run `python app.py` to re-seed.

---

## 📦 Dependencies

Only 3 packages needed:
- **Flask** — web framework
- **Flask-SQLAlchemy** — database ORM
- **Flask-Login** — session management
- All ML code is pure Python (no scikit-learn needed)
