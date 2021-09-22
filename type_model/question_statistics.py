def print_question_statistics(questions):
    all_texts = set()
    no_one_kotlin = set()

    subtype_yes = 0
    has_member_yes = 0
    has_parameter_yes = 0
    returns_yes = 0

    subtype_no = 0
    has_member_no = 0
    has_parameter_no = 0
    returns_no = 0

    for ql in questions:
        for q in ql:
            all_texts.add(q.text)

            if 'kotlin' not in q.text:
                no_one_kotlin.add(q.text)

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

    print('total questions  : %d' % len(all_texts))
    print('no one `kotlin.*`: %d' % len(no_one_kotlin))

    print('subtype       %d %d' % (subtype_yes, subtype_no))
    print('has_member    %d %d' % (has_member_yes, has_member_no))
    print('has_parameter %d %d' % (has_parameter_yes, has_parameter_no))
    print('returns       %d %d' % (returns_yes, returns_no))
