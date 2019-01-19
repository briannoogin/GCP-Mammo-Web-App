import * as React from "react";

import Titles from "./components/Titles";
import BoxCard from "./components/BoxCard"

const API_KEY = "3585775f387b0d0cba6c5b3dc41b8167";

const auth = require('google-auth-library');
class App extends React.Component 
{
  state = {
    temperature: 10,
    city: 'Plano',
    country: 'United States',
    humidity: 3,
    description: 'hot af',
    error: "Please enter the values."
  }
  getWeather()
  {
    console.log('hi')
  }
  render() 
  {
    return (
      <div>
        <div className="wrapper">
          <div className="main">
            <div className="container-fluid">
              <div className="row">
                <div className="col-sm-5 col-xs-5 title-container">
                  <Titles/>
                </div>
                <div className="col-sm-7 col-xs-7 form-container">
                  {/* <Form getWeather={this.getWeather} />
                  <Weather 
                    temperature={this.state.temperature} 
                    humidity={this.state.humidity}
                    city={this.state.city}
                    country={this.state.country}
                    description={this.state.description}
                    error={this.state.error} 
                  /> */}
                  <BoxCard></BoxCard>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  async predict(img:Array<Number>)
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
};

export default App;