import pandas as pd
import matplotlib.pyplot as plt
import wandb

def classify_question_type(question: str):
    if "누구" in question:
        return "who"
    elif "어디" in question:
        return "where"
    elif "언제" in question:
        return "when"
    elif "어떤" in question:
        return "what"
    elif "왜" in question:
        return "why"
    elif "어떻게" in question:
        return "how"
    elif "몇" in question:
        return "how many"
    else:
        return "others"
        
def retrieval_check(df, weight, check_passage_cnt, term):

    # Apply the function to your DataFrame
    df['question_type'] = df['question'].apply(classify_question_type)
    q_dict = {'who': 0, 'who_total' : 0, 
                'where': 0, 'where_total' : 0,
                'when': 0, 'when_total' : 0,
                'what': 0, 'what_total' : 0,
                'why': 0, 'why_total' : 0,
                'how': 0, 'how_total' : 0,
                'how many': 0, 'how many_total' : 0,
                'others': 0, 'others_total' : 0,
                'total' : len(df)}


    total_score = 0

    # df['context'] 를 줄 별로 나눠서 df["context"] 와 대조 비교
    context_columns = [
        "context" + str(i) for i in range(1, check_passage_cnt + 1)
    ]

    for i in range(term, check_passage_cnt + 1, term):
        df[
            "correct_count_" + str(i)
        ] = 0  # 'correct_count'라는 새로운 컬럼을 만들어서 0으로 초기화합니다.

    passages = [0 for i in range(check_passage_cnt // term)]

    df['ground_truth_rank'] = [0 for i in range(len(df))]

    for i in range(len(df)):
        for col in context_columns:
            if df.iloc[i][col] == df.iloc[i]["original_context"]:
                for u in range(term, check_passage_cnt + 1, term):

                    if int(col[7:]) <= u:
                        passages[int(u // term) - 1] += 1
                        total_score += weight[int(u // term) - 1]     

                        if u <= 10:
                            q_dict[df.iloc[i]['question_type']] += 1
                        q_dict[df.iloc[i]['question_type'] + '_total'] += 1   
                        df.iloc[i]['ground_truth_rank'] = int(col[7:])
                        break
                    else:
                        continue
                df.iloc[i]['ground_truth_rank'] = check_passage_cnt+10



    
    # 1
    # Set the data
    passages.append(len(df))
    data = passages
    labels = ["p" + str(i) for i in range(term, check_passage_cnt + 1, term)]
    labels.append("all")

    # Create bar chart
    fig, ax = plt.subplots()
    ax.set_title("Number of Correct Retrievals")
    ax.set_xlabel("Context columns")
    ax.set_ylabel("Number of correct retrievals")
    print(labels)
    print(data)
    bars = ax.bar(labels, data)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            yval,
            ha="center",
            va="bottom",
        )
    ax.set_title("Number of Correct Retrievals")
    ax.set_xlabel("Context columns")
    ax.set_ylabel("Number of correct retrievals")

    wandb.log({"retrieval_bar_chart": wandb.Image(fig)})

    # total_score 정규화
    print("total_score", total_score)
    print("len(df)*weight[0]", len(df) * weight[0])
    wandb.log({"retrieval_score": total_score / (len(df) * weight[0])})

    

    # 2 
    # question type 별 정답률
    print(q_dict)
    # dict to ax
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("Question Type")
    ax.set_xlabel("Question Type")
    ax.set_ylabel("Number of correct retrievals")
    bars = ax.bar(q_dict.keys(), q_dict.values())
    plt.xticks(rotation=55)
    plt.rc('xtick', labelsize=10) 

    for bar, label in zip(bars, q_dict.keys()):
        if 'total' in label :
            bar.set_color('r')  # 바의 색상을 red로 설정. 원하는 다른 색상으로 변경 가능
        if 'total' == label:
            bar.set_color('g')


    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            yval,
            ha="center",
            va="bottom",
        )
    wandb.log({"question_type": wandb.Image(fig)})
