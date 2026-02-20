test_probs = lgb_final.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'id': test_ids,
    'Heart Disease': test_probs
})

submission.to_csv("submission.csv", index=False)
