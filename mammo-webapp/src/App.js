import React, { Component } from 'react';
import './css/App.css';
import ImageFileDrop from './ImageFileDrop'
class App extends Component 
{

  render() {
    return (
      <div className="App">
         <header className="App-header">
          <h1>Welcome to my mammogram classifier web app!</h1>
        </header>
        <ImageFileDrop></ImageFileDrop>
      </div>
    );
  }
}

export default App;
