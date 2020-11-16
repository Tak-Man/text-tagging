from datetime import datetime
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import tools as tools

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

import matplotlib.pyplot as plt

start_time = datetime.now()
print("process started at {}".format(start_time.strftime('%Y-%m-%d-%H:%M:%S')))

RND_ST = 2857

tagged_data_df, label_col_names = tools.prepare_data(source_file="./data/tagged_data.csv",
                                                     min_tag_count=200,
                                                     max_tag_count=800,
                                                     keep_tag_types=["Label"])

tagged_data_df.to_csv("./test/test.csv", sep="\t")
print(tagged_data_df[tagged_data_df["Review_Link"] == "/dstv-multichoice/reviews/-2501141"])

num_labels = len(label_col_names)
plot_rows = 2
plot_cols = int(round(num_labels / plot_rows, 0))  # len(label_col_names)
print("plot_rows - {}, plot_cols - {}".format(plot_rows, plot_cols))

# Plot single labels
plot_positions = []
count = 0
for row in range(0, plot_rows):
    for col in range(0, plot_cols):
        plot_positions.append((row, col))
        count += 1
        if count >= num_labels:
            break

print("plot_positions :", plot_positions)
fig1, ax1 = plt.subplots(plot_rows, plot_cols, sharey=True, figsize=(12, 8))

for col_name, (row, col) in zip(label_col_names, plot_positions):
    values = tagged_data_df[col_name].value_counts()
    ax1[row, col].bar(values.index, values.values)
    ax1[row, col].set_title(col_name, fontsize=10)

# plt.tight_layout()
fig1.suptitle("Label Counts", size=14, y=0.98)

plt.show()
plt.close()

# Plot label power set
power_set = tagged_data_df[label_col_names].apply(lambda row: "".join(row.values.astype(str)), axis=1)
print("power_set :")
print(power_set)
power_set_counts = power_set.value_counts().sort_values(ascending=True)
print("power_set_counts :")
print(power_set_counts)
plt.figure(figsize=(8, 13))
plt.barh(power_set_counts.index, power_set_counts.values, color="green")
plt.title("Label Power Sets Counts")

plt.show()
plt.close()

X = tagged_data_df["Review_Text"]
y = tagged_data_df.iloc[:, 2:]
print("X :")
print(X[0:2])
print("X.shape :", X.shape)
print("y.shape :", y.shape)
print("y :", y[:2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RND_ST)
# vect = TfidfVectorizer(ngram_range=(1, 3))#, max_features=1000)
# X_train_vect = vect.fit_transform(X_train)
# print("X_train_vect.shape :", X_train_vect.shape)
#
# # reducer = SelectKBest(k=1000)
# # X_train_vect = reducer.fit_transform(X_train_vect, y_train)
# # print("X_train_vect.shape :", X_train_vect.shape)
# # X_test_vect = reducer.transform(vect.transform(X_text))
#
# X_test_vect = vect.transform(X_text)
# estimator = OneVsRestClassifier(estimator=LogisticRegression(random_state=RND_ST))
# estimator.fit(X_train_vect, y_train)
# test_predict = estimator.predict(X_test_vect)
# print("y_test :")
# print(y_test[:5])
# print("test_predict :")
# print(test_predict[:5])


# score_hamming_loss = hamming_loss(y_test, test_predict)
# accuracy_score = accuracy_score(y_test, test_predict)
# precision_score = precision_score(y_test, test_predict, average="weighted")
# recall_score = recall_score(y_test, test_predict, average="weighted")
# roc_auc_score = roc_auc_score(y_test, test_predict) #, average="weighted")
# f1_score = f1_score(y_test, test_predict, average="weighted")
# print("score_hamming_loss :", score_hamming_loss)
# print("accuracy_score :", accuracy_score)
# print("precision_score :", precision_score)
# print("recall_score :", recall_score)
# print("roc_auc_score :", roc_auc_score)
# print("f1_score :", f1_score)
# print()

# for label_idx in range(0, num_labels):
#     label_str = label_col_names[label_idx]
#     temp_pred = test_predict[:, label_idx]
#     temp_test = y_test.values[:, label_idx]
#     temp_score_hamming_loss = hamming_loss(temp_test, temp_pred)
#     temp_accuracy_score = accuracy_score(temp_test, temp_pred)
#     temp_precision_score = precision_score(temp_test, temp_pred)
#     temp_recall_score = recall_score(temp_test, temp_pred)
#     temp_roc_auc_score = roc_auc_score(temp_test, temp_pred)
#     temp_f1_score = roc_auc_score(temp_test, temp_pred)
#
#     print("Score for-'{}'".format(label_str))
#     print("temp_score_hamming_loss :", temp_score_hamming_loss)
#     print("temp_accuracy_score :", temp_accuracy_score)
#     print("temp_precision_score :", temp_precision_score)
#     print("temp_recall_score :", temp_recall_score)
#     print("temp_roc_auc_score :", temp_roc_auc_score)
#     print("temp_f1_score :", temp_f1_score)
#     print()










# Build a pipeline and use grid search to get good parameters
pipe_1 = Pipeline([("vec", TfidfVectorizer()),
                   ("red", SelectKBest()),
                   ("clf", OneVsRestClassifier(estimator=LogisticRegression()))])
param_grid_1 = [{
                 "vec__ngram_range": [(1, 3)],
                 "vec__stop_words": ["english"],
                 # "vec__max_features": [500, 1000],
                 "red__score_func": [chi2],
                 "red__k": [200, 500, 1000],
                 "clf__estimator__C": [1, 10],
                 "clf__estimator__random_state": [RND_ST]
                # {"": }
                 }]
gs = GridSearchCV(pipe_1, param_grid_1, scoring="roc_auc", cv=5)
gs.fit(X_train, y_train)
best_estimator = gs.best_estimator_
best_params = gs.best_params_

print("best_params :", best_params)

test_predict = best_estimator.predict(X_test)
test_predict_proba = best_estimator.predict_proba(X_test)
print("y_test :")
print(y_test[:5])
print("test_predict :")
print(test_predict[:5])

# score_hamming_loss = hamming_loss(y_test, test_predict)
# accuracy_score = accuracy_score(y_test, test_predict)
# precision_score = precision_score(y_test, test_predict, average="weighted")
# recall_score = recall_score(y_test, test_predict, average="weighted")
# roc_auc_score = roc_auc_score(y_test, test_predict) #, average="weighted")
# f1_score = f1_score(y_test, test_predict, average="weighted")
# print("score_hamming_loss :", score_hamming_loss)
# print("accuracy_score :", accuracy_score)
# print("precision_score :", precision_score)
# print("recall_score :", recall_score)
# print("roc_auc_score :", roc_auc_score)
# print("f1_score :", f1_score)
# print()


# Plot ROC AUC Curves
fig2, ax2 = plt.subplots(plot_rows, plot_cols, sharey=True, sharex=True, figsize=(12, 8))

for col_name, col_idx, (row, col) in zip(label_col_names, range(0, num_labels), plot_positions):
    # label_str = label_col_names[col_idx]
    temp_pred = test_predict[:, col_idx]
    temp_pred_proba = test_predict_proba[:, col_idx]
    temp_test = y_test.values[:, col_idx]
    temp_score_hamming_loss = hamming_loss(temp_test, temp_pred)
    temp_accuracy_score = accuracy_score(temp_test, temp_pred)
    temp_precision_score = precision_score(temp_test, temp_pred)
    temp_recall_score = recall_score(temp_test, temp_pred)
    temp_roc_auc_score = roc_auc_score(temp_test, temp_pred)
    temp_roc_auc_score_proba = roc_auc_score(temp_test, temp_pred_proba)
    temp_f1_score = roc_auc_score(temp_test, temp_pred)

    print("Score for-'{}'".format(col_name))
    print("temp_score_hamming_loss :", temp_score_hamming_loss)
    print("temp_accuracy_score :", temp_accuracy_score)
    print("temp_precision_score :", temp_precision_score)
    print("temp_recall_score :", temp_recall_score)
    print("temp_roc_auc_score :", temp_roc_auc_score)
    print("temp_roc_auc_score_proba :", temp_roc_auc_score_proba)
    print("temp_f1_score :", temp_f1_score)
    print()

    values = tagged_data_df[col_name].value_counts()
    ax2[row, col].set_title(col_name, fontsize=8)

    false_positive_rate, true_positive_rate, _ = roc_curve(temp_test, temp_pred_proba)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    ax2[row, col].plot(false_positive_rate, true_positive_rate, color="darkgreen",
                       linewidth=3, label="Estimator")
    ax2[row, col].fill_between(x=false_positive_rate, y1=true_positive_rate, color="green", alpha=0.25)  # 'slateblue'
    ax2[row, col].text(0.05, 0.95, 'AUC={0:0.2f}'.format(temp_roc_auc_score_proba))
    ax2[row, col].plot([0, 1], [0, 1], 'k--', linewidth=3, color='grey', label="No skill")

    ax2[row, col].set_xlim([0.0, 1.0])
    ax2[row, col].set_ylim([0.0, 1.05])
    ax2[row, col].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax2[row, col].set_title(label="'" + col_name + "'", fontsize=10.0)
    ax2[row, col].legend(loc="lower right")

fig2.suptitle("ROC AUC Plots", size=16, y=0.98)
# plt.tight_layout()
plt.show()
plt.close()

# plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# plt.suptitle(title, size=18, y=adj_no_rows * 0.245) # 0.96
# plt.text(-1.1, 1.2 * adj_no_rows, optimised_parameters_str, fontsize=8.0, wrap=True)

end_time = datetime.now()
print("Process ended at {}".format(end_time.strftime('%Y-%m-%d-%H:%M:%S')))

duration = end_time - start_time
print("Duration :", duration)

