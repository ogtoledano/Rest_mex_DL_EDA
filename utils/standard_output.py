import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score, accuracy_score, mean_absolute_error

def make_txt_file_out(task,X,model, device, url_file):
    f = open(url_file+"output_sentiment.txt", "a")
    f.write(score_sentiment(X,model,device))
    f.close()


def make_txt_file_out_two_task(X, model1, model2, device, url_file):
    f = open(url_file+"output_sentiment.txt", "a")
    f.write(score_sentiment_two_task(X,model1, model2, device))
    f.close()


def score_sentiment(X, model, device):
    model.to(device)
    model.eval()
    output = ""
    with torch.no_grad():
        for i,data in enumerate(X):
            x_test = torch.from_numpy(data['features']).type(torch.LongTensor).to(device)
            x_test = torch.reshape(x_test, (1, x_test.shape[0]))
            prob = model(x_test)
            _, predicted = torch.max(prob.data, 1)
            output += "\"recommendation\"\t\"Usuario\"{}\t{}\n".format(i+1, predicted.cpu().numpy()[0]+1)

    return output


def score_sentiment_two_task(X, model1,model2, device):
    model1.to(device)
    model1.eval()

    model2.to(device)
    model2.eval()

    atractions = {0: "Hotel", 1: "Restaurant", 2: "Attractive"}

    predictions = []
    labels_ref = []

    output = ""
    with torch.no_grad():
        for i,data in enumerate(X):
            input_ids = data['source_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels=data['labels'].to(device)
            # input_ids = torch.from_numpy(input_ids).type(torch.LongTensor).to(device)
            input_ids = torch.reshape(input_ids, (1, input_ids.shape[0]))
            attention_mask = torch.reshape(attention_mask, (1, attention_mask.shape[0]))

            outputs1 = model1(input_ids=input_ids,attention_mask=attention_mask)
            predicted1 = torch.argmax(outputs1.logits, dim=-1)

            outputs2 = model2(input_ids=input_ids,attention_mask=attention_mask)
            predicted2 = torch.argmax(outputs2.logits, dim=-1)

            predictions.extend(predicted1.cpu().numpy())
            labels_ref.extend(predicted2.cpu().numpy())

            output += "\"sentiment\"\t\"{}\"\t\"{}\"\t\"{}\"\n".format(i+1, predicted1.cpu().numpy()[0]+1, atractions[predicted2.cpu().numpy()[0]])

    accuracy = accuracy_score(labels_ref, predictions)
    mae = mean_absolute_error(labels_ref, predictions)
    macro_f1 = f1_score(labels_ref, predictions, average='macro')

    print("acc: {}, mae: {}, macrof1:{}".format(accuracy,mae,macro_f1))
    output+="acc: {}, mae: {}, macrof1:{}".format(accuracy,mae,macro_f1)
    return output