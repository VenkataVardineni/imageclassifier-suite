import React from 'react';
import './PredictionResult.css';

function PredictionResult({ prediction }) {
  const { predicted_class, confidence, top3_predictions } = prediction;

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return '#4caf50';
    if (conf >= 0.6) return '#ff9800';
    return '#f44336';
  };

  const getClassEmoji = (className) => {
    const emojis = {
      airplane: '✈️',
      automobile: '🚗',
      bird: '🐦',
      cat: '🐱',
      deer: '🦌',
      dog: '🐶',
      frog: '🐸',
      horse: '🐴',
      ship: '🚢',
      truck: '🚚'
    };
    return emojis[className.toLowerCase()] || '📦';
  };

  return (
    <div className="prediction-result">
      <h2>Prediction Results</h2>
      
      <div className="main-prediction">
        <div className="prediction-header">
          <span className="class-emoji">{getClassEmoji(predicted_class)}</span>
          <h3 className="predicted-class">{predicted_class}</h3>
        </div>
        <div className="confidence-bar-container">
          <div className="confidence-label">
            Confidence: {(confidence * 100).toFixed(2)}%
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${confidence * 100}%`,
                backgroundColor: getConfidenceColor(confidence)
              }}
            />
          </div>
        </div>
      </div>

      <div className="top-predictions">
        <h4>Top 3 Predictions</h4>
        <div className="predictions-list">
          {top3_predictions.map((pred, index) => (
            <div key={index} className="prediction-item">
              <div className="prediction-rank">#{index + 1}</div>
              <div className="prediction-details">
                <div className="prediction-class-row">
                  <span className="prediction-emoji">{getClassEmoji(pred.class)}</span>
                  <span className="prediction-class-name">{pred.class}</span>
                </div>
                <div className="prediction-confidence-row">
                  <div className="mini-confidence-bar">
                    <div
                      className="mini-confidence-fill"
                      style={{
                        width: `${pred.confidence * 100}%`,
                        backgroundColor: getConfidenceColor(pred.confidence)
                      }}
                    />
                  </div>
                  <span className="prediction-percentage">
                    {(pred.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default PredictionResult;

