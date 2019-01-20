import React, { Component } from 'react';
import FileDrop from'react-file-drop'
import '../css/ImageFileDrop.css'
var jpeg = require('jpeg-js');

// const path = require('path');
class ImageFileDrop extends Component 
{
    state = {
      img:Int8Array,
    }
    handleDrop = (files:FileList, event:any) => 
    {
        // get file from list of files
        const file = files[0];
        let array_reader = new FileReader();
        let data_reader = new FileReader();
        // read as img url
        data_reader.readAsDataURL(file);
        // function reads image data to get dimensions 
        data_reader.onload = function()
        {
          // read the file as an array buffer
          array_reader.readAsArrayBuffer(file);
        }
        // function that reads the array data from array buffer
        array_reader.onload = function()
        {
          let img:Uint8Array = jpeg.decode(array_reader.result);
          let json:object = {
            'img':img
          };
          const URL = 'http://127.0.0.1:5000/api'
          const OPTIONS: RequestInit = {
            body:JSON.stringify(json),
            method:'POST',
            mode:'cors',
            headers:{
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          };
          fetch(URL,OPTIONS)  
          .then(res =>{res.json().then(prediction => console.log(prediction))})
          .catch(error =>{console.log(error)})
        };
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