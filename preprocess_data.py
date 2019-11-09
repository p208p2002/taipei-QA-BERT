
from core import AnsDic,QuestionDic

def make_ans_dic(answers):
    ansdic = AnsDic(answers)
    print("全部答案:",len(ansdic))
    print("全部答案種類:",ansdic.types)
    
    # 測試轉換
    a_id = ansdic.to_id('臺北市信義區公所')
    a_text = ansdic.to_text(a_id)
    assert a_text == '臺北市信義區公所'
    assert ansdic.to_id(a_text) == a_id

    return ansdic

def make_question_dic(quetsions):
    return QuestionDic(quetsions)
    

if __name__ == "__main__":
    with open('Taipei_QA_new.txt','r',encoding='utf-8') as f:
        data = f.read()
    qa_pairs = data.split("\n")

    questions = []
    answers = []
    for qa_pair in qa_pairs:
        qa_pair = qa_pair.split()
        try:
            a,q = qa_pair
            questions.append(q)
            answers.append(a)
        except:
            continue
    
    assert len(answers) == len(questions)
    
    ans_dic = make_ans_dic(answers)
    question_dic = make_question_dic(questions)
    print(question_dic.to_id('104年8月15日施行之道路交通管理處罰條例條文為何?'))
    
