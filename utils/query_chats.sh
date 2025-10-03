# Find all chats with a specific display name
sqlite3 ~/Library/Messages/chat.db "
SELECT ROWID, chat_identifier, display_name
FROM chat
WHERE display_name = 'name'
ORDER BY ROWID;
"

# Check timestamps of recent messages in a chat
sqlite3 ~/Library/Messages/chat.db "
SELECT datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') AS msg_time,
       h.id AS sender,
       m.text              
FROM message m 
JOIN handle h ON m.handle_id = h.ROWID
JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
WHERE cmj.chat_id = 1234
ORDER BY m.date DESC
LIMIT 5;
"

# create working copy of chat.db
cp ~/Library/Messages/chat.db ./utils/chat_copy.db

# extract messages to txt file
python ./utils/extract_messages.py > ./imessage_input.txt