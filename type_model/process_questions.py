import random

from type_model.generate_questions import QuestionSample


def process_questions(questions):
    make_unique(questions)
    equalize_kt_types_with_user_types(questions)
    equalize_yes_no(questions)


def make_unique(questions):
    unique = set()
    for list_id in range(len(questions)):
        new_list = []
        for q_id in range(len(questions[list_id])):
            q = questions[list_id][q_id]

            if q.text not in unique:
                unique.add(q.text)
                new_list.append(q)
        questions[list_id] = new_list


def equalize_yes_no(questions):
    subtype_yes = 0
    has_member_yes = 0
    has_parameter_yes = 0
    returns_yes = 0

    subtype_no = 0
    has_member_no = 0
    has_parameter_no = 0
    returns_no = 0

    for list_id in range(len(questions)):
        for q_id in range(len(questions[list_id])):
            q: QuestionSample = questions[list_id][q_id]

            if q.true_answer == 1.0:
                if 0 <= q.question_id <= 4:
                    subtype_yes += 1
                elif 5 <= q.question_id <= 8:
                    has_member_yes += 1
                elif 9 <= q.question_id <= 11:
                    has_parameter_yes += 1
                elif 12 <= q.question_id <= 14:
                    returns_yes += 1
            else:
                if 0 <= q.question_id <= 4:
                    subtype_no += 1
                elif 5 <= q.question_id <= 8:
                    has_member_no += 1
                elif 9 <= q.question_id <= 11:
                    has_parameter_no += 1
                elif 12 <= q.question_id <= 14:
                    returns_no += 1

    p_subtype_yes = min(subtype_yes, subtype_no) / subtype_yes
    p_subtype_no = min(subtype_yes, subtype_no) / subtype_no

    p_has_member_yes = min(has_member_yes, has_member_no) / has_member_yes
    p_has_member_no = min(has_member_yes, has_member_no) / has_member_no

    p_has_parameter_yes = min(has_parameter_yes, has_parameter_no) / has_parameter_yes
    p_has_parameter_no = min(has_parameter_yes, has_parameter_no) / has_parameter_no

    p_returns_yes = min(returns_yes, returns_no) / returns_yes
    p_returns_no = min(returns_yes, returns_no) / returns_no

    for list_id in range(len(questions)):
        new_list = []
        for q_id in range(len(questions[list_id])):
            q: QuestionSample = questions[list_id][q_id]

            probability = 0.0
            if q.true_answer == 1.0:
                if 0 <= q.question_id <= 4:
                    probability = p_subtype_yes
                elif 5 <= q.question_id <= 8:
                    probability = p_has_member_yes
                elif 9 <= q.question_id <= 11:
                    probability = p_has_parameter_yes
                elif 12 <= q.question_id <= 14:
                    probability = p_returns_yes
            else:
                if 0 <= q.question_id <= 4:
                    probability = p_subtype_no
                elif 5 <= q.question_id <= 8:
                    probability = p_has_member_no
                elif 9 <= q.question_id <= 11:
                    probability = p_has_parameter_no
                elif 12 <= q.question_id <= 14:
                    probability = p_returns_no

            if random.random() < probability:
                new_list.append(q)
        questions[list_id] = new_list


def equalize_kt_types_with_user_types(questions):
    for list_id in range(len(questions)):
        new_list = []
        for q_id in range(len(questions[list_id])):
            q: QuestionSample = questions[list_id][q_id]

            if 'kotlin' in q.text:
                if random.random() < 0.1:
                    new_list.append(q)
            else:
                new_list.append(q)
        questions[list_id] = new_list
