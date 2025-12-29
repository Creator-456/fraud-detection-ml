"""
Fraud Detection Predictor
Real-time fraud detection using trained models
Author: Dev Narayan Chaudhary
"""

import numpy as np

class FraudDetector:
    """Simple fraud detector for demonstration"""
    
    def predict_fraud(self, transaction):
        """
        Predict fraud probability for a transaction
        
        Args:
            transaction (dict): Transaction details with keys:
                - amount: Transaction amount
                - transaction_hour: Hour of day (0-23)
                - customer_age_days: Account age in days
                - distance_from_home: Distance from home in km
            
        Returns:
            dict: Prediction results including probability, prediction, risk level
        """
        # Simple rule-based scoring for demo
        risk_score = 0
        
        # High amount transactions are riskier
        if transaction.get('amount', 0) > 1000:
            risk_score += 0.3
        
        # Late night transactions are riskier
        if transaction.get('transaction_hour', 12) >= 22:
            risk_score += 0.25
        
        # New customers are riskier
        if transaction.get('customer_age_days', 365) < 90:
            risk_score += 0.25
        
        # Transactions far from home are riskier
        if transaction.get('distance_from_home', 0) > 100:
            risk_score += 0.2
        
        is_fraud = 1 if risk_score > 0.5 else 0
        
        return {
            'fraud_probability': risk_score,
            'is_fraud': is_fraud,
            'risk_level': self._get_risk_level(risk_score)
        }
    
    def _get_risk_level(self, score):
        """Get risk level from score"""
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MEDIUM"
        elif score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"


# Demo usage
if __name__ == "__main__":
    detector = FraudDetector()
    
    # Test transactions
    transactions = [
        {
            'name': 'Small daytime purchase',
            'amount': 45.99,
            'transaction_hour': 14,
            'customer_age_days': 730,
            'distance_from_home': 5
        },
        {
            'name': 'Large night transaction',
            'amount': 1850,
            'transaction_hour': 23,
            'customer_age_days': 45,
            'distance_from_home': 250
        }
    ]
    
    print("="*60)
    print("FRAUD DETECTION DEMO")
    print("="*60)
    
    for txn in transactions:
        name = txn.pop('name')
        result = detector.predict_fraud(txn)
        
        print(f"\n{name}:")
        print(f"  Amount: ${txn['amount']}")
        print(f"  Fraud Probability: {result['fraud_probability']:.2%}")
        print(f"  Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"  Risk Level: {result['risk_level']}")
