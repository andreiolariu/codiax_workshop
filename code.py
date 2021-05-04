import pickle
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


with open('matches.pickle', 'rb') as f:
  matches = pickle.loads(f.read())
matches = matches[:-1]

# matches2.append({'member_id': 'Andrei',
#  'request_id': 'Codiax',
#  'mismatch': True,
#  'auto_generated': False,
#  'message': 'remove this from the matches list'})

# with open('matches.pickle', 'wb') as f:
#   f.write(pickle.dumps(matches2))

matched_pairs = set((d['member_id'], d['request_id']) for d in matches)
member_ids = list({d['member_id'] for d in matches})
request_ids = list({d['request_id'] for d in matches})


auto_wrong = [d for d in matches if d['mismatch'] is True and \
    d['auto_generated'] is True]
auto_right = [d for d in matches if d['mismatch'] is False and \
    d['auto_generated'] is True]
manual_right = [d for d in matches if d['mismatch'] is False and \
    d['auto_generated'] is False]
random_wrong = []
while True:
  member_id = member_ids[random.randint(0, len(member_ids) - 1)]
  request_id = request_ids[random.randint(0, len(request_ids) - 1)]
  if (member_id, request_id) in matched_pairs:
    continue
  random_wrong.append({
      'member_id': member_id,
      'request_id': request_id,
  })
  matched_pairs.add((member_id, request_id))

  if len(random_wrong) == 120000:
    break


X_matches = auto_wrong + auto_right + manual_right + random_wrong
y = [0] * len(auto_wrong) + [1] * len(auto_right) + [1] * len(manual_right) + \
    [0] * len(random_wrong)
baseline_pred = [1] * len(auto_wrong) + [1] * len(auto_right) + \
    [0] * len(manual_right) + [0] * len(random_wrong)

y = np.array(y)
baseline_pred = np.array(baseline_pred)


with open('encodings.pickle', 'rb') as f:
  member_encodings, request_encodings = pickle.loads(f.read())

X = []
for row in X_matches:
  m_enc = member_encodings[row['member_id']]
  r_enc = request_encodings[row['request_id']]
  # X.append(member_encodings[row['member_id']] * \
  #     request_encodings[row['request_id']])
  # X.append(np.abs(member_encodings[row['member_id']] - \
  #     request_encodings[row['request_id']]))
  X.append(np.hstack((m_enc, r_enc, m_enc * r_enc, np.abs(m_enc - r_enc))))

X = np.array(X)


X_train, X_test, y_train, y_test, _, bp_test = \
    train_test_split(X, y, baseline_pred, random_state=42, test_size=1000)

model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, \
    max_depth=3)
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)
# print('baseline:', sum(bp_test == y_test)/len(preds))
# print('new model:', sum(preds == y_test)/len(preds))


from sklearn import metrics

print('baseline:', metrics.roc_auc_score(y_test, bp_test))
print('new model:', metrics.roc_auc_score(y_test, preds[:,1]))


# 10 - 3 - 79.2
# 10 - 6 - 79.8
# 20 - 3 - 79.4
# 20 - 6 - 80.9
# 200 - 6 - 85.7
# 500 - 7 - 86.9