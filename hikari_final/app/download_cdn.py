import urllib.request
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

print("Downloading tailwindcss...")
req1 = urllib.request.Request("https://cdn.tailwindcss.com", headers=headers)
with urllib.request.urlopen(req1) as response, open("../static/tailwind.js", 'wb') as out_file:
    out_file.write(response.read())

print("Downloading socket.io...")
req2 = urllib.request.Request("https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js", headers=headers)
with urllib.request.urlopen(req2) as response, open("../static/socket.io.min.js", 'wb') as out_file:
    out_file.write(response.read())

print("Downloads complete.")
