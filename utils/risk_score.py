def risk_score(probability: float) -> str:
    
    if probability < 0.6:
        return "Allow"
    

    elif 0.6 <= probability < 0.85:
        return "Allow and Monitor"
    

    else:
        return "Block"
