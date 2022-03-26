import torch


def make_txt_file_out(task,X,model, device, url_file):
    f = open(url_file+"output_sentiment.txt", "a")
    f.write(score_sentiment(X,model,device))
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

    output = ""
    with torch.no_grad():
        for i,data in enumerate(X):
            x_test = torch.from_numpy(data['features']).type(torch.LongTensor).to(device)
            x_test = torch.reshape(x_test, (1, x_test.shape[0]))

            prob1 = model1(x_test)
            _, predicted1 = torch.max(prob1.data, 1)

            prob2 = model2(x_test)
            _, predicted2 = torch.max(prob2.data, 1)

            output += "\"sentiment\"\t\"{}\"\t\"{}\"\t\"{}\"\n".format(i+1, predicted1.cpu().numpy()[0]+1, atractions[predicted1.cpu().numpy()[0]])

    return output