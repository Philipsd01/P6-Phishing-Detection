// Code.gs

// Use the endpoint confirmed from your ngrok forward
const LOCAL_API_ENDPOINT =
  "https://5e80-2a02-aa7-4118-a28c-7471-dc2c-724c-8e10.ngrok-free.app/predict";

/**
 * Callback function for contextual trigger when reading a message.
 * @param {Object} e The event object passed by the trigger.
 * @return {Card[]} Array containing the card(s) to display.
 */
function buildPhishingCheckCard(e) {
  try {
    // --- 1. Get Email Details ---
    const messageId = e.gmail.messageId;
    const accessToken = e.gmail.accessToken;
    GmailApp.setCurrentMessageAccessToken(accessToken); // Necessary to use GmailApp service

    const message = GmailApp.getMessageById(messageId);
    const subject = message.getSubject();
    const body = message.getPlainBody(); // Using plain text body

    // --- 2. Prepare API Payload ---
    // Combine subject and body, remove any CR/LF characters by replacing them with a space, and trim any extra whitespace
    const rawText = (subject + " " + body).replace(/[\r\n]+/g, " ").trim();

    // Create payload with a single field for raw text.
    const payload = {
      raw_text: rawText,
    };

    console.log("Payload: " + JSON.stringify(payload)); // for debugging

    // --- 3. Set API Call Options ---
    const options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify(payload),
      muteHttpExceptions: true,
    };

    // --- 4. Call Backend API ---
    console.log("Calling API: " + LOCAL_API_ENDPOINT); // Log the call for debugging
    const response = UrlFetchApp.fetch(LOCAL_API_ENDPOINT, options);
    const responseCode = response.getResponseCode();
    const responseBody = response.getContentText();
    console.log("API Response Code: " + responseCode);

    // --- 5. Process API Response ---
    let resultText = "Checking...";
    let isPhishing = false;

    if (responseCode === 200) {
      try {
        const predictionResult = JSON.parse(responseBody);
        if (predictionResult.hasOwnProperty("is_phishing")) {
          isPhishing = predictionResult.is_phishing;
          resultText = isPhishing
            ? "⚠️ Potential Phishing Detected!"
            : "✅ Looks Safe";
        } else {
          console.error(
            "API response missing 'is_phishing' key. Body: " + responseBody
          );
          resultText = "Error: Invalid API response format.";
        }
      } catch (err) {
        console.error(
          "Failed to parse API JSON response: " +
            err +
            ". Body: " +
            responseBody
        );
        resultText = "Error: Could not parse prediction result.";
      }
    } else {
      console.error(
        "API request failed. Code: " + responseCode + ", Body: " + responseBody
      );
      resultText =
        "Error: Could not contact detection service. (Code: " +
        responseCode +
        ")";
      if (responseCode === 404) {
        resultText += " Endpoint not found.";
      } else if (responseCode >= 500) {
        resultText += " Server error.";
      }
    }

    // --- 6. Determine Icon URL ---
    let iconUrl = isPhishing
      ? "https://gds.baguette.engineering/icons/clear.png" // Warning icon
      : "https://gds.baguette.engineering/icons/check.png"; // Check icon

    // --- 7. Build Card Header ---
    const cardHeader = CardService.newCardHeader()
      .setTitle("Phishing Scan Result")
      .setImageUrl(iconUrl);

    // --- 8. Build Card Sections and Widgets ---
    const cardSection = CardService.newCardSection()
      .addWidget(CardService.newTextParagraph().setText(resultText))
      .addWidget(
        CardService.newTextParagraph().setText(`<b>Subject:</b> ${subject}`)
      );

    // --- 9. Build the Card ---
    const cardBuilder = CardService.newCardBuilder()
      .setHeader(cardHeader)
      .addSection(cardSection);

    const finalCard = cardBuilder.build();

    // --- 10. Return the Card ---
    return [finalCard];
  } catch (error) {
    console.error("Error within buildPhishingCheckCard function: " + error);
    console.error("Stack: " + error.stack);

    const errorCard = CardService.newCardBuilder()
      .setHeader(CardService.newCardHeader().setTitle("Add-on Error"))
      .addSection(
        CardService.newCardSection()
          .addWidget(
            CardService.newTextParagraph().setText(
              "An unexpected error occurred while running the add-on. Check logs for details."
            )
          )
          .addWidget(CardService.newTextParagraph().setText(error.toString()))
      )
      .build();
    return [errorCard];
  }
}
