# Import
import numpy as np
import tensorflow as tf
import transformers
from flask import Flask, request

# Tokenizer
print('Loading tokenizer...')
tokenizer = transformers.AlbertTokenizer.from_pretrained('./tokenizer_config', do_lower_case=True)
def tokenize(tokenizer, data):
  input_ids = list()
  attention_masks = list()
  for text in data:
    processed_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_attention_mask=True)
    input_ids.append(processed_text['input_ids'])
    attention_masks.append(processed_text['attention_mask'])
  return np.array(input_ids).astype('int32'), np.array(attention_masks).astype('int32')

# Load model
print('Creating model...')
x_ids = tf.keras.layers.Input(128, dtype='int32')
x_masks = tf.keras.layers.Input(128, dtype='int32')
y = transformers.TFAlbertModel(transformers.PretrainedConfig.from_pretrained('./transformer_config'))([x_ids, x_masks])
y_a = y[1]
y_b = y[0]
y_b = tf.squeeze(y_b[:, -1:, :], axis=1)
y = tf.keras.layers.Concatenate()([y_a, y_b])
y = tf.keras.layers.Dense(32, activation='relu')(y)
y = tf.keras.layers.Dropout(0.2)(y)
y = tf.keras.layers.Dense(1, activation='sigmoid')(y)
model = tf.keras.models.Model(inputs=[x_ids, x_masks], outputs=y)
print('Loading weights...')
model.load_weights('./sentiment_analysis_lite_weights.h5')

# Sample inference
def infer(text, model, tokenize):
  ids, masks = tokenize(tokenizer, [text])
  prob = model.predict([ids, masks])[0][0]
  sent = 'Positive' if prob > 0.5 else 'Negative'
  return sent, float('{:.4f}'.format(prob))
  
# Flask app
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  text = request.get_json(force=True)['text']
  if text is None:
    return {'success': False, 'message': 'Text field is empty'}
  sent, prob = infer(text, model, tokenize)
  return {'success': True, 'sentiment': sent, 'polarity_score': prob}

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001, debug=False)