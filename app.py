"""
AutoGrade — Flask Automated Answer Grading System
Teacher login | Student login | ML grading | Progress tracking
"""
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import json, re, math
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'autograde-flask-secret-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autograde.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
lm = LoginManager(app)
lm.login_view = 'login'

# ── MODELS ────────────────────────────────────────────────────────────────────

class User(UserMixin, db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    username   = db.Column(db.String(80),  unique=True, nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(80),  default='')
    last_name  = db.Column(db.String(80),  default='')
    role       = db.Column(db.String(10),  default='student')
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)
    submissions = db.relationship('Submission', backref='student', lazy=True)

    def full_name(self):
        n = f"{self.first_name} {self.last_name}".strip()
        return n if n else self.username
    def is_teacher(self): return self.role == 'teacher'
    def is_student(self): return self.role == 'student'

@lm.user_loader
def load_user(uid): return User.query.get(int(uid))


class Subject(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    name      = db.Column(db.String(100), unique=True, nullable=False)
    questions = db.relationship('Question', backref='subject', lazy=True)
    exams     = db.relationship('Exam',     backref='subject', lazy=True)


exam_questions = db.Table('exam_questions',
    db.Column('exam_id',     db.Integer, db.ForeignKey('exam.id')),
    db.Column('question_id', db.Integer, db.ForeignKey('question.id')),
)


class Question(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    subject_id       = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    creator_id       = db.Column(db.Integer, db.ForeignKey('user.id'),    nullable=False)
    question_text    = db.Column(db.Text, nullable=False)
    reference_answer = db.Column(db.Text, nullable=False)
    max_marks        = db.Column(db.Integer, default=10)
    is_active        = db.Column(db.Boolean, default=True)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    creator          = db.relationship('User', backref='questions')


class Exam(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    subject_id  = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    creator_id  = db.Column(db.Integer, db.ForeignKey('user.id'),    nullable=False)
    is_active   = db.Column(db.Boolean, default=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    questions   = db.relationship('Question', secondary=exam_questions, lazy='subquery')
    submissions = db.relationship('Submission', backref='exam', lazy=True)
    creator     = db.relationship('User', backref='exams')

    def total_marks(self):
        return sum(q.max_marks for q in self.questions)


class Submission(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    student_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    exam_id      = db.Column(db.Integer, db.ForeignKey('exam.id'), nullable=False)
    total_score  = db.Column(db.Float,   default=0)
    total_marks  = db.Column(db.Integer, default=0)
    percentage   = db.Column(db.Float,   default=0)
    grade        = db.Column(db.String(3), default='F')
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    answers      = db.relationship('AnswerSubmission', backref='submission', lazy=True)


class AnswerSubmission(db.Model):
    id                  = db.Column(db.Integer, primary_key=True)
    submission_id       = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=False)
    question_id         = db.Column(db.Integer, db.ForeignKey('question.id'),   nullable=False)
    student_answer      = db.Column(db.Text, default='')
    score               = db.Column(db.Float, default=0)
    percentage          = db.Column(db.Float, default=0)
    grade               = db.Column(db.String(3), default='F')
    feedback            = db.Column(db.Text, default='')
    semantic_similarity = db.Column(db.Float, default=0)
    keyword_coverage    = db.Column(db.Float, default=0)
    length_adequacy     = db.Column(db.Float, default=0)
    question            = db.relationship('Question')


# ── ML GRADER ─────────────────────────────────────────────────────────────────

STOPWORDS = {"a","an","the","is","it","in","on","of","to","and","or","but","for","with","at","by","from",
             "as","are","was","were","be","been","being","have","has","had","do","does","did","will","would",
             "could","should","may","might","this","that","these","those","i","we","you","he","she","they",
             "its","our","your","his","her","their","also","so","if","then","than","when","where","which",
             "who","what","how","why","not","no","very","just","more","each","all","any","both","few",
             "most","other","some","into","through","before","after","above","below","between","out",
             "off","over","under","again","during"}

def _pre(t):
    t = re.sub(r'[^\w\s]',' ',t.lower())
    return [x for x in t.split() if x not in STOPWORDS and len(x)>1]

def _stem(w):
    for s in ['ing','tion','ness','ment','able','ible','ed','ly','er','est','al','ive']:
        if w.endswith(s) and len(w)-len(s)>=3: return w[:-len(s)]
    return w

def _tf(toks):
    c=Counter(toks); tot=len(toks) or 1
    return {w:v/tot for w,v in c.items()}

def _cos(a,b):
    ks=set(a)|set(b)
    dot=sum(a.get(k,0)*b.get(k,0) for k in ks)
    ma=math.sqrt(sum(v**2 for v in a.values()))
    mb=math.sqrt(sum(v**2 for v in b.values()))
    return dot/(ma*mb) if ma and mb else 0.0

def _grade(pct):
    if pct>=90: return 'A+'
    if pct>=80: return 'A'
    if pct>=70: return 'B+'
    if pct>=60: return 'B'
    if pct>=50: return 'C'
    if pct>=40: return 'D'
    return 'F'

def grade_answer(ref, ans, max_marks=10):
    if not ans or not ans.strip():
        return dict(score=0,percentage=0.0,grade='F',feedback='No answer provided.',
                    semantic_similarity=0,keyword_coverage=0,length_adequacy=0)
    rt=[_stem(t) for t in _pre(ref)]
    st=[_stem(t) for t in _pre(ans)]
    cos=_cos(_tf(rt),_tf(st))
    kw=len(set(rt)&set(st))/len(set(rt)) if rt else 0
    lp=1.0 if len(st)>=len(rt)*0.5 else (0.5+len(st)/len(rt)) if rt else 1.0
    sim=(cos*0.55+kw*0.45)*lp
    if sim>0.15: sim=0.15+(sim-0.15)*1.25
    sim=min(max(sim,0.0),1.0)
    pct=round(sim*100,1)
    msgs=[]
    if pct>=80: msgs.append("Excellent! Key concepts covered well.")
    elif pct>=60: msgs.append("Good attempt. Most important points addressed.")
    elif pct>=40: msgs.append("Partial credit. Some key points missing.")
    else: msgs.append("Answer needs significant improvement.")
    missing=list(set(rt)-set(st))[:4]
    if missing and pct<80: msgs.append(f"Consider including: {', '.join(missing)}.")
    if kw<0.4: msgs.append("Use more subject-specific terminology.")
    if len(st)<len(rt)*0.3: msgs.append("Your answer is too brief — elaborate more.")
    return dict(score=round(sim*max_marks,2),percentage=pct,grade=_grade(pct),
                feedback=" ".join(msgs),
                semantic_similarity=round(cos*100,1),
                keyword_coverage=round(kw*100,1),
                length_adequacy=round(lp*100,1))


# ── DECORATORS ────────────────────────────────────────────────────────────────

def teacher_required(f):
    @wraps(f)
    @login_required
    def decorated(*a,**k):
        if not current_user.is_teacher():
            flash('Teacher access required.','error')
            return redirect(url_for('student_dashboard'))
        return f(*a,**k)
    return decorated

def student_required(f):
    @wraps(f)
    @login_required
    def decorated(*a,**k):
        if not current_user.is_student():
            return redirect(url_for('teacher_dashboard'))
        return f(*a,**k)
    return decorated


# ── AUTH ROUTES ───────────────────────────────────────────────────────────────

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('teacher_dashboard' if current_user.is_teacher() else 'student_dashboard'))
    return render_template('auth/home.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        uname=request.form.get('username','').strip()
        email=request.form.get('email','').strip()
        pwd=request.form.get('password','')
        pwd2=request.form.get('password2','')
        role=request.form.get('role','student')
        errors=[]
        if not uname: errors.append('Username required.')
        if User.query.filter_by(username=uname).first(): errors.append('Username taken.')
        if User.query.filter_by(email=email).first(): errors.append('Email already registered.')
        if len(pwd)<4: errors.append('Password min 4 chars.')
        if pwd!=pwd2: errors.append('Passwords do not match.')
        if errors:
            for e in errors: flash(e,'error')
            return render_template('auth/register.html',form=request.form)
        u=User(username=uname,email=email,password=generate_password_hash(pwd),
               first_name=request.form.get('first_name','').strip(),
               last_name=request.form.get('last_name','').strip(),role=role)
        db.session.add(u); db.session.commit()
        login_user(u)
        flash(f'Welcome, {u.full_name()}!','success')
        return redirect(url_for('teacher_dashboard' if role=='teacher' else 'student_dashboard'))
    return render_template('auth/register.html',form={})

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    error=None
    if request.method=='POST':
        u=User.query.filter_by(username=request.form.get('username','')).first()
        if u and check_password_hash(u.password,request.form.get('password','')):
            login_user(u)
            return redirect(url_for('teacher_dashboard' if u.is_teacher() else 'student_dashboard'))
        error='Invalid username or password.'
    return render_template('auth/login.html',error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))


# ── TEACHER ROUTES ────────────────────────────────────────────────────────────

@app.route('/teacher')
@teacher_required
def teacher_dashboard():
    questions=Question.query.filter_by(creator_id=current_user.id).order_by(Question.created_at.desc()).all()
    exams=Exam.query.filter_by(creator_id=current_user.id).order_by(Exam.created_at.desc()).all()
    recent=(Submission.query.join(Exam).filter(Exam.creator_id==current_user.id)
            .order_by(Submission.submitted_at.desc()).limit(10).all())
    total_students=(db.session.query(Submission.student_id).join(Exam)
                    .filter(Exam.creator_id==current_user.id).distinct().count())
    return render_template('teacher/dashboard.html',questions=questions,exams=exams,
                           recent=recent,total_students=total_students,
                           subjects=Subject.query.all())

@app.route('/teacher/question/add', methods=['GET','POST'])
@teacher_required
def add_question():
    if request.method=='POST':
        sname=request.form.get('subject_name','').strip()
        qt=request.form.get('question_text','').strip()
        ra=request.form.get('reference_answer','').strip()
        mm=int(request.form.get('max_marks',10))
        if not sname or not qt or not ra:
            flash('All fields are required.','error')
            return render_template('teacher/add_question.html',subjects=Subject.query.all(),form=request.form)
        subj=Subject.query.filter(Subject.name.ilike(sname)).first()
        if not subj:
            subj=Subject(name=sname); db.session.add(subj); db.session.flush()
        q=Question(subject_id=subj.id,creator_id=current_user.id,
                   question_text=qt,reference_answer=ra,max_marks=mm)
        db.session.add(q); db.session.commit()
        flash(f'Question added to "{subj.name}"!','success')
        return redirect(url_for('teacher_dashboard'))
    return render_template('teacher/add_question.html',subjects=Subject.query.all(),form={})

@app.route('/teacher/question/<int:qid>/edit', methods=['GET','POST'])
@teacher_required
def edit_question(qid):
    q=Question.query.filter_by(id=qid,creator_id=current_user.id).first_or_404()
    if request.method=='POST':
        sname=request.form.get('subject_name','').strip()
        subj=Subject.query.filter(Subject.name.ilike(sname)).first()
        if not subj:
            subj=Subject(name=sname); db.session.add(subj); db.session.flush()
        q.subject_id=subj.id
        q.question_text=request.form.get('question_text','').strip()
        q.reference_answer=request.form.get('reference_answer','').strip()
        q.max_marks=int(request.form.get('max_marks',10))
        db.session.commit()
        flash('Question updated!','success')
        return redirect(url_for('teacher_dashboard'))
    return render_template('teacher/add_question.html',subjects=Subject.query.all(),
                           form=q,edit=True,question=q)

@app.route('/teacher/question/<int:qid>/delete', methods=['POST'])
@teacher_required
def delete_question(qid):
    q=Question.query.filter_by(id=qid,creator_id=current_user.id).first_or_404()
    db.session.delete(q); db.session.commit()
    flash('Question deleted.','success')
    return redirect(url_for('teacher_dashboard'))

@app.route('/teacher/exam/create', methods=['GET','POST'])
@teacher_required
def create_exam():

    if request.method == 'POST':

        title = request.form.get('title','').strip()

        # selected subject from dropdown
        subj_id = request.form.get('subject_id', type=int)

        # new subject typed by teacher
        subject_name = request.form.get('subject_name','').strip()

        q_ids = request.form.getlist('questions', type=int)

        is_active = 'is_active' in request.form


        # if teacher typed a new subject
        if subject_name:

            subject = Subject.query.filter(
                Subject.name.ilike(subject_name)
            ).first()

            if not subject:
                subject = Subject(name=subject_name)
                db.session.add(subject)
                db.session.flush()

            subj_id = subject.id


        if not title or not subj_id or not q_ids:
            flash('Fill all fields and select at least one question.','error')

        else:

            exam = Exam(
                title=title,
                subject_id=subj_id,
                creator_id=current_user.id,
                is_active=is_active
            )

            db.session.add(exam)
            db.session.flush()

            exam.questions = Question.query.filter(
                Question.id.in_(q_ids)
            ).all()

            db.session.commit()

            flash(f'Exam "{title}" created!','success')

            return redirect(url_for('teacher_dashboard'))


    return render_template(
        'teacher/create_exam.html',
        subjects=Subject.query.all(),
        questions=Question.query.filter_by(
            creator_id=current_user.id,
            is_active=True
        ).all()
    )
@app.route('/teacher/exam/<int:exam_id>/results')
@teacher_required
def exam_results(exam_id):
    exam=Exam.query.filter_by(id=exam_id,creator_id=current_user.id).first_or_404()
    subs=Submission.query.filter_by(exam_id=exam_id).order_by(Submission.submitted_at.desc()).all()
    avg=round(sum(s.percentage for s in subs)/len(subs),1) if subs else 0
    return render_template('teacher/exam_results.html',exam=exam,submissions=subs,avg_pct=avg)

@app.route('/teacher/submission/<int:sub_id>')
@teacher_required
def student_detail(sub_id):
    sub=(Submission.query.join(Exam)
         .filter(Submission.id==sub_id,Exam.creator_id==current_user.id).first_or_404())
    return render_template('teacher/student_detail.html',sub=sub)


# ── STUDENT ROUTES ────────────────────────────────────────────────────────────

@app.route('/student')
@student_required
def student_dashboard():
    attempted={s.exam_id for s in current_user.submissions}
    available=Exam.query.filter_by(is_active=True).filter(~Exam.id.in_(attempted or {-1})).all()
    subs=Submission.query.filter_by(student_id=current_user.id).order_by(Submission.submitted_at.desc()).all()
    avg=round(sum(s.percentage for s in subs)/len(subs),1) if subs else 0
    return render_template('student/dashboard.html',available=available,submissions=subs,avg_pct=avg)

@app.route('/student/exam/<int:exam_id>')
@student_required
def take_exam(exam_id):
    exam=Exam.query.filter_by(id=exam_id,is_active=True).first_or_404()
    if Submission.query.filter_by(student_id=current_user.id,exam_id=exam_id).first():
        flash('You have already submitted this exam.','warning')
        return redirect(url_for('student_dashboard'))
    return render_template('student/take_exam.html',exam=exam)

@app.route('/student/exam/<int:exam_id>/submit', methods=['POST'])
@student_required
def submit_exam(exam_id):
    exam=Exam.query.filter_by(id=exam_id,is_active=True).first_or_404()
    if Submission.query.filter_by(student_id=current_user.id,exam_id=exam_id).first():
        return redirect(url_for('student_dashboard'))
    sub=Submission(student_id=current_user.id,exam_id=exam_id)
    db.session.add(sub); db.session.flush()
    total_score=total_marks=0
    for q in exam.questions:
        if not q.is_active: continue
        ans=request.form.get(f'answer_{q.id}','').strip()
        r=grade_answer(q.reference_answer,ans,q.max_marks)
        db.session.add(AnswerSubmission(
            submission_id=sub.id,question_id=q.id,student_answer=ans,
            score=r['score'],percentage=r['percentage'],grade=r['grade'],
            feedback=r['feedback'],
            semantic_similarity=r['semantic_similarity'],
            keyword_coverage=r['keyword_coverage'],
            length_adequacy=r['length_adequacy']))
        total_score+=r['score']; total_marks+=q.max_marks
    pct=round((total_score/total_marks)*100,1) if total_marks else 0
    sub.total_score=round(total_score,2); sub.total_marks=total_marks
    sub.percentage=pct; sub.grade=_grade(pct)
    db.session.commit()
    return redirect(url_for('view_result',sub_id=sub.id))

@app.route('/student/result/<int:sub_id>')
@student_required
def view_result(sub_id):
    sub=Submission.query.filter_by(id=sub_id,student_id=current_user.id).first_or_404()
    return render_template('student/result.html',sub=sub)

@app.route('/student/progress')
@student_required
def progress():
    subs=Submission.query.filter_by(student_id=current_user.id).order_by(Submission.submitted_at.asc()).all()
    chart=[{'exam':s.exam.title[:22],'pct':s.percentage,'grade':s.grade} for s in subs]
    avg=round(sum(s.percentage for s in subs)/len(subs),1) if subs else 0
    best=max(subs,key=lambda s:s.percentage) if subs else None
    return render_template('student/progress.html',submissions=subs,
                           chart_data=json.dumps(chart),avg_pct=avg,best=best)


# ── AJAX ──────────────────────────────────────────────────────────────────────

@app.route('/api/live-grade', methods=['POST'])
@login_required
def live_grade():
    data=request.get_json(force=True)
    q=Question.query.get(data.get('question_id'))
    if not q: return jsonify({'error':'Not found'}),404
    return jsonify(grade_answer(q.reference_answer,data.get('answer',''),q.max_marks))


# ── SEED ──────────────────────────────────────────────────────────────────────

def seed():
    if User.query.filter_by(username='teacher_demo').first(): return
    teacher=User(username='teacher_demo',email='teacher@autograde.com',
                 password=generate_password_hash('teacher123'),
                 first_name='Prof. Ravi',last_name='Kumar',role='teacher')
    db.session.add(teacher)
    students=[]
    for uname,fname,lname in [('student_asha','Asha','Sharma'),
                               ('student_rahul','Rahul','Verma'),
                               ('student_priya','Priya','Nair')]:
        s=User(username=uname,email=f'{uname}@autograde.com',
               password=generate_password_hash(uname.split('_')[1]+'123'),
               first_name=fname,last_name=lname,role='student')
        db.session.add(s); students.append(s)
    db.session.flush()

    SEED=[
        ("Computer Science",[
            ("What is Machine Learning? Explain its three main types.",
             "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. The three main types are Supervised Learning where the model is trained on labelled data, Unsupervised Learning where the model finds hidden patterns in unlabelled data, and Reinforcement Learning where an agent learns by interacting with an environment and receiving rewards or penalties.",10),
            ("Explain the four pillars of Object-Oriented Programming.",
             "Object Oriented Programming organises software around objects. The four pillars are Encapsulation which bundles data and methods inside a class restricting direct access, Abstraction which hides complex implementation and exposes only essential features, Inheritance which allows a child class to acquire properties from a parent class enabling code reuse, and Polymorphism which allows objects of different classes to be treated as a common type through method overriding and overloading.",10),
            ("What is a Binary Search Tree? Describe its properties and operations.",
             "A Binary Search Tree is a node-based binary tree where each node has at most two children. The left subtree contains nodes with keys less than the node key and the right subtree contains nodes with greater keys. Main operations include Search which traverses left or right in O log n average time, Insertion which adds a new leaf at the correct position, and Deletion which handles leaf nodes one child or two children using the in-order successor.",10),
        ]),
        ("Networking",[
            ("Explain the OSI model and describe each of its seven layers.",
             "The OSI model standardises communication into seven layers. Physical layer transmits raw bits. Data Link handles node-to-node transfer and error detection using MAC addresses. Network manages routing using IP addresses. Transport provides end-to-end communication using TCP and UDP. Session establishes manages and terminates sessions. Presentation handles encryption decryption and data format translation. Application provides services to users via HTTP FTP and SMTP.",10),
            ("What is the difference between TCP and UDP?",
             "TCP is connection-oriented and establishes connection via three-way handshake guaranteeing reliable ordered error-checked delivery using acknowledgements and retransmission. TCP is used for web browsing email and file transfer where data integrity is critical. UDP is connectionless with no handshaking sending datagrams without guaranteeing delivery order or error checking making it faster. UDP is preferred for live streaming online gaming VoIP and DNS where speed matters more than reliability.",10),
        ]),
        ("Database Management",[
            ("What is database normalisation? Explain 1NF 2NF and 3NF.",
             "Database normalisation reduces data redundancy and improves integrity. First Normal Form requires all columns to contain atomic indivisible values with no repeating groups. Second Normal Form requires 1NF and every non-key attribute must be fully functionally dependent on the entire primary key with no partial dependencies. Third Normal Form requires 2NF and all non-key attributes must be directly dependent on the primary key with no transitive dependencies.",10),
            ("Explain the ACID properties of database transactions.",
             "ACID describes four properties. Atomicity means a transaction is all or nothing preventing partial updates. Consistency ensures the database moves from one valid state to another maintaining all integrity constraints. Isolation ensures concurrent transactions execute as if sequential preventing interference. Durability means committed changes are permanently saved even after failures typically ensured through write-ahead logging and backups.",10),
        ]),
        ("Mathematics",[
            ("What is a derivative in calculus? Explain power rule chain rule and product rule.",
             "A derivative measures the instantaneous rate of change of a function and represents the slope of the tangent line at any point. Power Rule: derivative of x to n is n times x to n minus 1. Product Rule: derivative of u times v is u prime v plus u v prime. Chain Rule: derivative of g of h of x is g prime of h times h prime used for composite functions. Derivatives are fundamental in optimisation physics and engineering.",10),
            ("Explain matrices: what they are how to multiply them and what the determinant represents.",
             "A matrix is a rectangular array of numbers in rows and columns. Matrix multiplication of A m by n and B n by p produces C m by p where element i j is the dot product of row i of A and column j of B. The number of columns in A must equal the number of rows in B and multiplication is not commutative. The determinant is a scalar value from a square matrix. For 2 by 2 matrix it is ad minus bc. Zero determinant means the matrix is singular non-invertible.",10),
        ]),
        ("Physics",[
            ("State and explain Newton's three laws of motion with real-world examples.",
             "Newton's Three Laws form the foundation of classical mechanics. First Law of Inertia: an object at rest stays at rest and in motion stays in motion unless acted on by a net external force. Example passenger lurches forward when car brakes. Second Law F equals ma: acceleration is proportional to net force and inversely proportional to mass. Third Law: every action has an equal and opposite reaction. Example rocket expels gases downward and thrust pushes it upward.",10),
            ("Explain the laws of thermodynamics.",
             "The Laws of Thermodynamics describe heat work and energy. Zeroth Law: systems in thermal equilibrium with a third are in equilibrium with each other forming the basis of thermometry. First Law conservation of energy: change in internal energy equals heat added minus work done. Second Law: entropy of an isolated system always increases and no heat engine can be 100 percent efficient. Third Law: as temperature approaches absolute zero entropy approaches minimum constant value.",10),
        ]),
        ("History",[
            ("What were the main causes and consequences of World War I?",
             "World War I 1914 to 1918 was caused by Militarism as nations built large armies creating an arms race, Alliances dividing Europe into Triple Alliance Germany Austria Hungary Italy and Triple Entente France Russia Britain, Imperialism through competition for colonies, and Nationalism especially in the Balkans. The immediate trigger was the assassination of Archduke Franz Ferdinand in Sarajevo 1914. Consequences included 17 million deaths collapse of four empires Treaty of Versailles League of Nations and seeds for World War II.",10),
            ("Describe the causes and key events of the Indian Independence Movement.",
             "The Indian Independence Movement was a mass struggle against British colonial rule culminating in independence on 15 August 1947. Causes included economic exploitation through taxation and deindustrialisation racial discrimination and political awakening with the Indian National Congress formed in 1885. Key events include Indian Rebellion of 1857 Swadeshi movement Gandhi Non Cooperation Movement 1920 Dandi Salt March 1930 Quit India Movement 1942 and Jallianwala Bagh Massacre 1919. The Indian Independence Act 1947 created India and Pakistan.",10),
        ]),
    ]

    sample_answers=[
        "Machine learning is a branch of AI where algorithms learn patterns from data. The three types are supervised learning using labelled data, unsupervised learning that finds hidden patterns, and reinforcement learning where an agent gets rewards.",
        "OOP has four pillars: encapsulation hides internal data, abstraction shows only essential details, inheritance lets child classes reuse parent code, and polymorphism allows different classes to be used through a common interface.",
        "A BST is a binary tree where left child has smaller keys and right child has larger keys. Operations include search insert and delete with O log n average complexity.",
        "Machine learning lets computers learn from data without explicit programming. There is supervised and unsupervised learning.",
        "OOP uses classes and objects. Inheritance allows code reuse.",
    ]

    all_qs=[]
    for subj_name,qs in SEED:
        subj=Subject(name=subj_name); db.session.add(subj); db.session.flush()
        for qt,ra,mm in qs:
            q=Question(subject_id=subj.id,creator_id=teacher.id,
                       question_text=qt,reference_answer=ra,max_marks=mm)
            db.session.add(q); all_qs.append(q)
    db.session.flush()

    subj_cs=Subject.query.filter_by(name="Computer Science").first()
    subj_ma=Subject.query.filter_by(name="Mathematics").first()
    subj_hi=Subject.query.filter_by(name="History").first()

    exams_data=[
        ("Computer Science Mid-Term",subj_cs,all_qs[0:3]),
        ("Science & Maths Combined",subj_ma,all_qs[8:12]),
        ("History & General Knowledge",subj_hi,all_qs[12:14]),
    ]
    exams=[]
    for title,subj,qs in exams_data:
        if subj:
            e=Exam(title=title,subject_id=subj.id,creator_id=teacher.id,is_active=True)
            e.questions=qs; db.session.add(e); exams.append(e)
    db.session.flush()

    if exams:
        cs_exam=exams[0]
        for i,student in enumerate(students):
            sub=Submission(student_id=student.id,exam_id=cs_exam.id)
            db.session.add(sub); db.session.flush()
            ts=tm=0
            for j,q in enumerate(cs_exam.questions):
                ans=sample_answers[(i*3+j)%len(sample_answers)]
                r=grade_answer(q.reference_answer,ans,q.max_marks)
                db.session.add(AnswerSubmission(
                    submission_id=sub.id,question_id=q.id,student_answer=ans,
                    score=r['score'],percentage=r['percentage'],grade=r['grade'],
                    feedback=r['feedback'],
                    semantic_similarity=r['semantic_similarity'],
                    keyword_coverage=r['keyword_coverage'],
                    length_adequacy=r['length_adequacy']))
                ts+=r['score']; tm+=q.max_marks
            pct=round((ts/tm)*100,1) if tm else 0
            sub.total_score=round(ts,2); sub.total_marks=tm
            sub.percentage=pct; sub.grade=_grade(pct)

    db.session.commit()
    print("✅ Seeded: teacher_demo/teacher123 | student_asha/asha123 | student_rahul/rahul123 | student_priya/priya123")

if __name__=='__main__':
    with app.app_context():
        db.create_all()
        seed()
    app.run(debug=True,port=5000)
