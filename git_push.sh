#!/bin/bash

echo "Aggiungo tutti i file modificati all'area di staging..."
git add .

echo "Fai il commit: inserisci un messaggio descrittivo:"
read commit_message

git commit -m "$commit_message"

echo "Invio le modifiche al repository remoto su GitHub..."
git push origin main

echo "Fatto! Modifiche inviate."
