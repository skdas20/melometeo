let userLocation = null;

// Request location permission
if ("geolocation" in navigator) {
  navigator.geolocation.getCurrentPosition(
    (position) => {
      userLocation = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude
      };
      console.log("User location obtained:", userLocation);
    },
    (error) => {
      console.error("Error obtaining location:", error);
      userLocation = null;
    }
  );
} else {
  console.error("Geolocation not supported.");
}

// Use matchMedia to check if mobile
const isMobile = window.matchMedia("(max-width: 768px)").matches;

// Hide loading screen and show main content after 3 seconds
setTimeout(() => {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('mainContent').style.display = 'block';
}, 3000);

// Common function: capture frame from a video element and send to /predict endpoint
function captureAndPredict(videoElement, resultElement) {
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL('image/jpeg');
  const payload = {
    image: dataURL,
    location: userLocation
  };
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => {
    console.log("Prediction response:", data);
    if (data.error) {
      resultElement.innerText = "Error: " + data.error;
    } else {
      resultElement.innerText = `Mood: ${data.mood} | Age: ${data.age}`;
    }
  })
  .catch(err => {
    console.error("Error during prediction:", err);
    resultElement.innerText = "Error during prediction";
  });
}

// Function to extract prediction from result text (for /play payload)
function extractPrediction(resultText) {
  let mood = "";
  let age = "";
  if (resultText.includes("Mood:") && resultText.includes("Age:")) {
    let parts = resultText.split("|");
    mood = parts[0].replace("Mood:", "").trim();
    age = parts[1].replace("Age:", "").trim();
  }
  return { mood, age };
}

// MOBILE SETUP
if (isMobile) {
  const mobileCamera = document.getElementById('mobileCamera');
  const resultMobile = document.getElementById('resultMobile');
  const mobilePlayBtn = document.getElementById('mobilePlayBtn');
  const quoteMobile = document.getElementById('quoteMobile');
  
  // Set a fixed quote for mobile
  quoteMobile.innerText = "Where words fail, music speaks.";
  
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      mobileCamera.srcObject = stream;
      // Once the camera is active, show the Play button
      mobilePlayBtn.style.display = 'block';
      // Start predictions every 2 seconds
      setInterval(() => {
        captureAndPredict(mobileCamera, resultMobile);
      }, 2000);
    })
    .catch(err => {
      console.error("Error accessing webcam:", err);
      resultMobile.innerText = "Error accessing webcam.";
    });
  
  // Mobile Play Button: trigger Spotify playback
  mobilePlayBtn.addEventListener('click', () => {
    const pred = extractPrediction(resultMobile.innerText);
    if (pred.mood && pred.age) {
      const payload = {
        mood: pred.mood,
        age: pred.age,
        location: userLocation
      };
      fetch('/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(response => response.json())
      .then(data => {
        console.log("Play response (mobile):", data);
        if (data.external_url) {
          window.open(data.external_url, '_blank');
        }
      })
      .catch(err => console.error("Error playing song (mobile):", err));
    }
  });
} else {
  // DESKTOP SETUP
  const desktopCamera = document.getElementById('desktopCamera');
  const resultDesktop = document.getElementById('resultDesktop');
  const tryItBtn = document.getElementById('tryItBtn');
  const quoteDesktop = document.getElementById('quoteDesktop');
  const desktopPlayBtn = document.getElementById('desktopPlayBtn');
  
  function fetchRandomQuote() {
    const quotes = [
      "Music expresses that which cannot be put into words.",
      "Where words fail, music speaks.",
      "Music is the literature of the heart.",
      "Without music, life would be a mistake."
    ];
    return quotes[Math.floor(Math.random() * quotes.length)];
  }
  quoteDesktop.innerText = fetchRandomQuote();
  
  // Initially, hide desktop camera feed and play button
  desktopCamera.style.display = 'none';
  desktopPlayBtn.style.display = 'none';
  
  // When "Try It!" is clicked, start the camera and predictions
  tryItBtn.addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        desktopCamera.srcObject = stream;
        desktopCamera.style.display = 'block';
        tryItBtn.disabled = true;
        tryItBtn.innerText = "Camera Active";
        // Show the desktop Play button once camera is active
        desktopPlayBtn.style.display = 'block';
        setInterval(() => {
          captureAndPredict(desktopCamera, resultDesktop);
        }, 2000);
      })
      .catch(err => {
        console.error("Error accessing webcam:", err);
        resultDesktop.innerText = "Error accessing webcam.";
      });
  });
  
  // Desktop Play Button: trigger Spotify playback on click
  desktopPlayBtn.addEventListener('click', () => {
    const pred = extractPrediction(resultDesktop.innerText);
    if (pred.mood && pred.age) {
      const payload = {
        mood: pred.mood,
        age: pred.age,
        location: userLocation
      };
      fetch('/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      .then(response => response.json())
      .then(data => {
        console.log("Play response (desktop):", data);
        if (data.external_url) {
          window.open(data.external_url, '_blank');
        }
      })
      .catch(err => console.error("Error playing song (desktop):", err));
    }
  });
  
  // Also allow "q" key press on desktop to trigger playback
  document.addEventListener('keydown', function(e) {
    if (e.key === 'q' || e.key === 'Q') {
      const pred = extractPrediction(resultDesktop.innerText);
      if (pred.mood && pred.age) {
        const payload = {
          mood: pred.mood,
          age: pred.age,
          location: userLocation
        };
        fetch('/play', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
          console.log("Play response via keypress (desktop):", data);
          if (data.external_url) {
            window.open(data.external_url, '_blank');
          }
        })
        .catch(err => console.error("Error playing song via keypress (desktop):", err));
      }
    }
  });
}
