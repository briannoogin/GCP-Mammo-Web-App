import React, { Component } from 'react';
import FileDrop from'react-file-drop'
import './css/ImageFileDrop.css'
class ImageFileDrop extends Component 
{
    handleDrop = (files, event) => {
        console.log(files, event);
      }
    render() {
      const styles = { border: '1px solid black', width: 600, color: 'black', padding: 20 };
      return (
          <div id="react-file-drop" style = {{styles}}>
            <FileDrop onDrop={this.handleDrop}>
              Drop some files here!
            </FileDrop>
          </div>
      );
    }
  }
  export default ImageFileDrop