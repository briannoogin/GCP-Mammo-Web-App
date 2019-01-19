import React, { Component } from 'react';
import FileDrop from'react-file-drop'
import '../css/ImageFileDrop.css'
class ImageFileDrop extends Component 
{
    handleDrop = (files, event) => 
    {
        // get file from list of files
        const file = files['0'];
        let array_reader = new FileReader();
        let data_reader = new FileReader();
        let img = undefined; 
        let img_array = undefined;
        // function reads image data to get dimensions 
        data_reader.onload = function()
        {
          img = new Image();
          //console.log(file)
          img.src = data_reader.result;
          // read the file as an array buffer
          array_reader.readAsArrayBuffer(file);
        }
        // function that reads the array data
        array_reader.onload = function()
        {
          // function converts array into 2d array based on the img_width
          function convertArray(array, img_width)
          {
            var img_array = [];
            // convert 1d array to 2d array with shape given by img_width
            while(array.length) 
            {
              img_array.push(array.splice(0,img_width));
            }
            // console.log(JSON.stringify(img_array));
            // console.log([img_array.length, img_array[0].length ]);
            return img_array;
            }
          // convert Uint8Array to normal Array so splice can be used 
          const array = Array.from(new Uint8Array(array_reader.result)); 
          // console.log(array_reader.result);
          // console.log(array.length);
          img_array = convertArray(array,img.width);
         
          // console.log([img_array.length,img_array[0].length]);
        };
        // read as img url
      data_reader.readAsDataURL(file);
    }
   
    async predict(img)
    {
      const link = 'https://ml.googleapis.com/v1/projects/MammoWebApp_Model:predict?key={YOUR_API_KEY}';
      const parameters = {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          predictions:img,
        })
      };
      const api_call = await fetch(link,parameters);
      const data = await api_call.json();
  } 
    render() 
    {
      //const styles = { border: '1px solid black', width: 600, color: 'black', padding: 20 };
      return (
          <div id="react-file-drop">
            <FileDrop onDrop={this.handleDrop}>
              Drag & Drop File
            </FileDrop>
          </div>
      );
    }
  }
  export default ImageFileDrop