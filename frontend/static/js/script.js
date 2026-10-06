// ---------------------------
// Preview selected image
// ---------------------------
document.addEventListener("change", function (e) {
    if (e.target.id === "fileInput") {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function (evt) {
          const img = document.createElement("img");
          img.src = evt.target.result;
          img.className = "preview-img";
          document.getElementById("preview").innerHTML = "";
          document.getElementById("preview").appendChild(img);
  
          // Save to session storage
          let uploads = JSON.parse(sessionStorage.getItem("uploads")) || [];
          uploads.push(evt.target.result);
          sessionStorage.setItem("uploads", JSON.stringify(uploads));
        };
        reader.readAsDataURL(file);
      }
    }
  });
  
  // ---------------------------
  // Show previous uploads (in results.html)
  // ---------------------------
  if (document.getElementById("resultsGallery")) {
    let uploads = JSON.parse(sessionStorage.getItem("uploads")) || [];
    uploads.forEach((src) => {
      const img = document.createElement("img");
      img.src = src;
      document.getElementById("resultsGallery").appendChild(img);
    });
  }
  
  // ---------------------------
  // Capture from camera (optional demo)
  // ---------------------------
  function openCamera() {
    alert("📷 Camera capture not implemented yet.");
  }
  
  // ---------------------------
  // Detect Coin (real backend call)
  // ---------------------------
  async function detectCoin() {
    const fileInput = document.getElementById("fileInput");
    const selectedFile = fileInput.files[0];
  
    const resultBox = document.getElementById("resultBox");
    resultBox.innerText = "Detecting... please wait.";
  
    if (!selectedFile) {
      alert("Please select a coin image first!");
      resultBox.innerText = "";
      return;
    }
  
    const formData = new FormData();
    formData.append("file", selectedFile);
  
    try {
      const res = await fetch("http://127.0.0.1:5000/detect_coin", {
        method: "POST",
        body: formData,
      });
  
      const data = await res.json();
      console.log("Result:", data);
  
      if (data.error) {
        resultBox.style.color = "red";
        resultBox.innerText = "❌ Error: " + data.error;
      } else {
        resultBox.style.color = "lime";
        resultBox.innerText =
          `✅ Coin: ${data.coin}\n` +
          `Confidence: ${(data.confidence * 100).toFixed(2)}%\n` +
          (data.year ? `Year: ${data.year}\n` : "") +
          (data.material ? `Material: ${data.material}` : "");
      }
    } catch (err) {
      console.error("Error:", err);
      resultBox.style.color = "red";
      resultBox.innerText =
        "Network or CORS error: Failed to fetch. Please check if Flask is running.";
    }
  }
  