import {Platform} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import model from './model.h5';

const loadModel = async () => {
  const modelPath = RNFS.MainBundlePath + '/model.h5';
  
    const modelJson = await RNFS.readFile(modelPath);
    const model = await tf.loadLayersModel({modelJson});
    
    return model;
  };