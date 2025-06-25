const { create, Client } = require('@open-wa/wa-automate');
const axios = require('axios');
const express = require('express');
const bodyParser = require('body-parser');

console.log('Starting WhatsApp bot...');

let globalClient = null;

create({
  headless: false, // Ensure browser opens for QR code
  qrTimeout: 0,    // Wait indefinitely for QR scan
  authTimeout: 0,  // Wait indefinitely for auth
  killProcessOnBrowserClose: true,
  useChrome: true,
  // Optionally, set a sessionId to force new session if needed
  // sessionId: 'WHATSAPP_SESSION',
}).then((client) => {
  console.log('WhatsApp client created. Waiting for QR code scan if not already authenticated...');
  globalClient = client;
  start(client);
  startExpressServer(client);
}).catch((err) => {
  console.error('Error creating WhatsApp client:', err);
  console.log('If you are not seeing a QR code, try deleting any session files in your bot directory and re-run this script.');
});

function start(client) {
  console.log('Bot is running. Send a WhatsApp message to your bot number to test.');
  client.onStateChanged((state) => {
    console.log('Client state changed:', state);
    if (state === 'CONFLICT' || state === 'UNLAUNCHED') client.forceRefocus();
  });

  client.onMessage(async (message) => {
    if (message.body && message.isGroupMsg === false) {
      console.log('Received message from', message.from, ':', message.body);
      // Send the message to your Python backend
      try {
        const response = await axios.post('http://localhost:5005/whatsapp', {
          from: message.from,
          body: message.body
        });
        // Send the response back to WhatsApp
        await client.sendText(message.from, response.data.reply);
        console.log('Replied to', message.from);
      } catch (err) {
        console.error('Error processing message:', err);
        await client.sendText(message.from, 'Error: Could not process your request.');
      }
    }
  });
}

function startExpressServer(client) {
  const app = express();
  app.use(bodyParser.json());

  // Endpoint for Python backend to send WhatsApp messages
  app.post('/send_message', async (req, res) => {
    const { to, message } = req.body;
    if (!to || !message) {
      return res.status(400).json({ error: 'Missing to or message' });
    }
    try {
      await client.sendText(to, message);
      console.log('Sent message to', to);
      res.json({ status: 'ok' });
    } catch (err) {
      console.error('Failed to send message to', to, err);
      res.status(500).json({ error: 'Failed to send message' });
    }
  });

  app.listen(3000, () => {
    console.log('Express server for WhatsApp bot listening on port 3000');
  });
} 