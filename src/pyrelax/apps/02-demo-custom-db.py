import pandas as pd

from pyrelax.engine import RelAlgEngine

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

students = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Carol", "David", "Eva"],
        "city": ["Vitória", "Rio", "Vitória", "São Paulo", "Rio"],
    }
)

courses = pd.DataFrame(
    {
        "id": [10, 20, 30],
        "name": ["Databases", "Algorithms", "AI"],
        "credits": [4, 4, 6],
    }
)

enrollments = pd.DataFrame(
    {
        "student_id": [1, 1, 2, 3, 4],
        "course_id": [10, 30, 10, 20, 30],
    }
)

db = {
    "Student": students,
    "Course": courses,
    "Enrollment": enrollments,
}

engine = RelAlgEngine(db)
print(engine.query("pi name sigma city = 'Vitória' Student"))
