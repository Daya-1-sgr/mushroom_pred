import pickle
from flask import Flask,render_template,request,jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
app=Flask(__name__)
model=pickle.load(open('log_reg_on_mushrooms.pkl','rb'))
label_encod=pickle.load(open('label_encoders_set.pkl','rb'))
column=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
attributes = {
    'cap-shape': {
        'bell': 'b',
        'conical': 'c',
        'convex': 'x',
        'flat': 'f',
        'knobbed': 'k',
        'sunken': 's'
    },
    'cap-surface': {
        'fibrous': 'f',
        'grooves': 'g',
        'scaly': 'y',
        'smooth': 's'
    },
    'cap-color': {
        'brown': 'n',
        'buff': 'b',
        'cinnamon': 'c',
        'gray': 'g',
        'green': 'r',
        'pink': 'p',
        'purple': 'u',
        'red': 'e',
        'white': 'w',
        'yellow': 'y'
    },
    'bruises': {
        'bruises': 't',
        'no': 'f'
    },
    'odor': {
        'almond': 'a',
        'anise': 'l',
        'creosote': 'c',
        'fishy': 'y',
        'foul': 'f',
        'musty': 'm',
        'none': 'n',
        'pungent': 'p',
        'spicy': 's'
    },

    'gill-spacing': {
        'close': 'c',
        'crowded': 'w',
        'distant': 'd'
    },
    'gill-size': {
        'broad': 'b',
        'narrow': 'n'
    },
    'gill-color': {
        'black': 'k',
        'brown': 'n',
        'buff': 'b',
        'chocolate': 'h',
        'gray': 'g',
        'green': 'r',
        'orange': 'o',
        'pink': 'p',
        'purple': 'u',
        'red': 'e',
        'white': 'w',
        'yellow': 'y'
    },
    'stalk-shape': {
        'enlarging': 'e',
        'tapering': 't'
    },
    'stalk-root': {
        'bulbous': 'b',
        'club': 'c',
        'cup': 'u',
        'equal': 'e',
        'rhizomorphs': 'z',
        'rooted': 'r',
        'missing': '?'
    },
    'stalk-surface-above-ring': {
        'fibrous': 'f',
        'scaly': 'y',
        'silky': 'k',
        'smooth': 's'
    },
    'stalk-surface-below-ring': {
        'fibrous': 'f',
        'scaly': 'y',
        'silky': 'k',
        'smooth': 's'
    },
    'stalk-color-above-ring': {
        'brown': 'n',
        'buff': 'b',
        'cinnamon': 'c',
        'gray': 'g',
        'orange': 'o',
        'pink': 'p',
        'red': 'e',
        'white': 'w',
        'yellow': 'y'
    },
    'stalk-color-below-ring': {
        'brown': 'n',
        'buff': 'b',
        'cinnamon': 'c',
        'gray': 'g',
        'orange': 'o',
        'pink': 'p',
        'red': 'e',
        'white': 'w',
        'yellow': 'y'
    },
    'veil-type': {
        'partial': 'p',
        'universal': 'u'
    },
    'veil-color': {
        'brown': 'n',
        'orange': 'o',
        'white': 'w',
        'yellow': 'y'
    },
    'ring-number': {
        'none': 'n',
        'one': 'o',
        'two': 't'
    },
    'ring-type': {
        'cobwebby': 'c',
        'evanescent': 'e',
        'flaring': 'f',
        'large': 'l',
        'none': 'n',
        'pendant': 'p',
        'sheathing': 's',
        'zone': 'z'
    },
    'spore-print-color': {
        'black': 'k',
        'brown': 'n',
        'buff': 'b',
        'chocolate': 'h',
        'green': 'r',
        'orange': 'o',
        'purple': 'u',
        'white': 'w',
        'yellow': 'y'
    },
    'population': {
        'abundant': 'a',
        'clustered': 'c',
        'numerous': 'n',
        'scattered': 's',
        'several': 'v',
        'solitary': 'y'
    },
    'habitat': {
        'grasses': 'g',
        'leaves': 'l',
        'meadows': 'm',
        'paths': 'p',
        'urban': 'u',
        'waste': 'w',
        'woods': 'd'
    }
}
@app.route('/')
def index():
    return render_template('index_css.html',attributes=attributes)

@app.route('/predict',methods=['POST'])
def predict():
    encoded_input=[]
    selections = {key: request.form.get(key) for key in attributes.keys()}
    for col in column:
        x=selections.get(col,None)
        y=label_encod.get(col,None)
        res=y.transform(np.array(x).reshape(1,-1))
        encoded_input.append(res)
    enc_arr=np.array(encoded_input).reshape(1,-1)
    pred=model.predict(enc_arr)
    result=label_encod['class'].inverse_transform(pred)
    if result==['e']:
       return 'mushroom is edible'
    else:
        return 'mushroom is inedible'
    # return selections
if __name__=='__main__':
    app.run(debug=True)