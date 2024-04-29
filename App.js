import React, { useEffect, useState } from 'react';
import { StyleSheet,View, Text, Image } from 'react-native';
import { Button, Input } from 'react-native-elements';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocossd from '@tensorflow-models/coco-ssd'
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as jpeg from 'jpeg-js';
import { fetch, bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as blazeface from '@tensorflow-models/blazeface';
import Svg, {Rect} from 'react-native-svg';
import * as Model from './utilits/modelLoader.js';


const App = () => {
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#fff',
      alignItems: 'center',
      justifyContent: 'center',
    },
  });
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState('');
  const [isEnabled,setIsEnabled] = useState(true)
  const [imageLink,setImageLink] = useState("https://raw.githubusercontent.com/ohyicong/masksdetection/master/dataset/without_mask/142.jpg")
  const [pickedImage, setPickedImage] = useState("https://raw.githubusercontent.com/ohyicong/masksdetection/master/dataset/without_mask/142.jpg");
  const [faces,setFaces]=useState([])
  const [faceDetector,setFaceDetector]=useState("")
  const [maskDetector,setMaskDetector]=useState("")
  
  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.cancelled) {
      setPickedImage(result.assets[0].uri);
    }
  };

  const classifyUsingMobilenet = async () => {
    try {
      // Load mobilenet.
      await tf.ready();
      const modelJson = await require("./assets/model/model.json");
      const modelWeight = await require("./assets/model/group1-shard.bin");
      const maskDetector = await tf.loadLayersModel(bundleResourceIO(modelJson,modelWeight));
      // const model = await mobilenet.load();
      // const model = await Model.loadModel();
      setIsTfReady(true);
      console.log("starting inference with picked image: " + pickedImage)

      // Convert image to tensor
      const imgB64 = await FileSystem.readAsStringAsync(pickedImage, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer)
      const imageTensor = decodeJpeg(raw);
      // Classify the tensor and show the result
      const prediction = await model.classify(imageTensor);
      if (prediction && prediction.length > 0) {
        setResult(
          `${prediction[0].className} (${prediction[0].probability.toFixed(3)})`
        );

        console.log("result : ", prediction[0].className)
      }
    } catch (err) {
      console.log(err);
    }
  };

  function imageToTensor(rawImageData){
    //Function to convert jpeg image to tensors
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    return tf.tensor3d(buffer, [height, width, 3]);
  }
  const getFaces = async() => {
    try{
      console.log("[+] Retrieving image from link :"+imageLink)
      const response = await fetch(imageLink, {}, { isBinary: true });
      const rawImageData = await response.arrayBuffer();
      const imageTensor = imageToTensor(rawImageData).resizeBilinear([224,224])
      const faces = await faceDetector.estimateFaces(imageTensor, false);
      var tempArray=[]
      //Loop through the available faces, check if the person is wearing a mask. 
      for (let i=0;i<faces.length;i++){
        let color = "red"
        let width = parseInt((faces[i].bottomRight[1] - faces[i].topLeft[1]))
        let height = parseInt((faces[i].bottomRight[0] - faces[i].topLeft[0]))
        let faceTensor=imageTensor.slice([parseInt(faces[i].topLeft[1]),parseInt(faces[i].topLeft[0]),0],[width,height,3])
        faceTensor = faceTensor.resizeBilinear([224,224]).reshape([1,224,224,3])
        let result = await maskDetector.predict(faceTensor).data()
        //if result[0]>result[1], the person is wearing a mask
        if(result[0]>result[1]){
          color="green"
        }
        tempArray.push({
          id:i,
          location:faces[i],
          color:color
        })
      }
      setFaces(tempArray)
      console.log("[+] Prediction Completed")
    }catch(error){
      console.log("[-] Unable to load image")
      console.log(error)
    }
    
  }
  // useEffect(() => {
  //   classifyUsingMobilenet()
  // }, [pickedImage]);
  useEffect(() => {
    async function loadModel(){
      //Wait for tensorflow module to be ready

      class L2 {

        static className = 'L2';
    
        constructor(config) {
           return tf.regularizers.l1l2(config)
        }
    }
    tf.serialization.registerClass(L2);
      console.log("[+] Application started")
      const tfReady = await tf.ready();

      //Replce model.json and group1-shard.bin with your own custom model
      console.log("[+] Loading custom mask detection model")
      const modelJson = await require("./assets/model/model.json");
      console.log("[+] Loading modelJson")
      const modelWeight = await require("./assets/model/group1-shard1of1.bin");
      console.log("[+] Loading modelWeight")
      try{
        const maskDetector = await tf.loadLayersModel(bundleResourceIO(modelJson, modelWeight));
      }catch(error){
        console.log("model : ",error )
      }
      
      console.log("[+] Loading pre-trained face detection model")
      // //Blazeface is a face detection model provided by Google

      const faceDetector =  await blazeface.load();
      console.log("[+] faceDetector")

      // //Assign model to variable
      setMaskDetector(maskDetector)
      setFaceDetector(faceDetector)
      // console.log("[+] Model Loaded")
    }
    loadModel()
  }, []); 
  return (
  //   <View style={styles.container}>
  //   <Input 
  //     placeholder="image link"
  //     onChangeText = {(inputText)=>{
  //       console.log(inputText)
  //       setImageLink(inputText)
  //       const elements= inputText.split(".")
  //       if(elements.slice(-1)[0]=="jpg" || elements.slice(-1)[0]=="jpeg"){
  //         setIsEnabled(true)
  //       }else{
  //         setIsEnabled(false)
  //       }
  //     }}
  //     value={imageLink}
  //     containerStyle={{height:40,fontSize:10,margin:15}} 
  //     inputContainerStyle={{borderRadius:10,borderWidth:1,paddingHorizontal:5}}  
  //     inputStyle={{fontSize:15}}
  //   />
  //   <View style={{marginBottom:20}}>
  //     <Image
  //       style={{width:224,height:224,borderWidth:2,borderColor:"black",resizeMode: "contain"}}
  //       source={{
  //         uri: imageLink
  //       }}
  //       PlaceholderContent={<View>No Image Found</View>}
  //     />
  //     <Svg height="224" width="224" style={{marginTop:-224}}>
  //       {
  //         faces.map((face)=>{
  //           return (
  //             <Rect
  //               key={face.id}
  //               x={face.location.topLeft[0]}
  //               y={face.location.topLeft[1]}
  //               width={(face.location.bottomRight[0] - face.location.topLeft[0])}
  //               height={(face.location.bottomRight[1] - face.location.topLeft[1])}
  //               stroke={face.color}
  //               strokeWidth="3"
  //               fill=""
  //             />
  //           )
  //         })
  //       }   
  //     </Svg>
  //   </View>
  //     <Button 
  //       title="Predict"
  //       onPress={()=>{getFaces()}}
  //       disabled={!isEnabled}
  //     />
  // </View>
    <View
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Image
        source={{ uri: pickedImage }}
        style={{ width: 200, height: 200, margin: 40 }}
      />
      {isTfReady && <Button
        title="Pick an image"
        onPress={pickImage}
      /> }
      <View style={{ width: '100%', height: 20 }} />
      {!isTfReady && <Text>Loading TFJS model...</Text>}
      {isTfReady && result === '' && <Text>Pick an image to classify!</Text>}
      {result !== '' && <Text>{result}</Text>}
      <Button 
          title="Predict"
          onPress={()=>{getFaces()}}
        />
    </View>
  );
};

export default App;