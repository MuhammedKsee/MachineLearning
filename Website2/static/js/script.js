document.addEventListener('DOMContentLoaded', function() {
    // DOM Elementleri - Tek bir yerde tanımlama
    const sidebar = document.querySelector('.sidebar');
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelectorAll('.nav-link');
    const submenuTriggers = document.querySelectorAll('.nav-group');
    const searchInput = document.querySelector('.search-input');
    const chatItems = document.querySelectorAll('.chat-item');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    const themeToggle = document.getElementById('themeToggle');
    const profileSettings = document.getElementById('profileSettings');

    // Tarih ve saat ekleme
    function updateDates() {
        const chatDates = document.querySelectorAll('.chat-date');
        const now = new Date();
        const formattedDate = now.toLocaleDateString('tr-TR', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        chatDates.forEach(dateElement => {
            dateElement.textContent = formattedDate;
        });
    }

    // Bot mesajı oluşturma
    function createBotMessage(message) {
        const botMessage = document.createElement('div');
        botMessage.className = 'message received';
        botMessage.innerHTML = `
            <div class="message-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
                </svg>
            </div>
            <div class="message-bubble">
                <div class="message-text">${message}</div>
                <div class="message-time">${new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
        `;
        return botMessage;
    }

    // Kullanıcı mesajı oluşturma
    function createUserMessage(message) {
        const userMessage = document.createElement('div');
        userMessage.className = 'message sent';
        userMessage.innerHTML = `
            <div class="message-bubble">
                <div class="message-text">${message}</div>
                <div class="message-time">${new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
        `;
        return userMessage;
    }

    // Dosya mesajı oluşturma (eğer varsa)
    function createFileMessage(file) {
        const fileMessage = document.createElement('div');
        fileMessage.className = 'message sent file-message';
        fileMessage.innerHTML = `
            <div class="message-bubble">
                <div class="file-info">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                    </svg>
                    <span>${file.name}</span>
                </div>
                <div class="message-time">${new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}</div>
            </div>
        `;
        return fileMessage;
    }

    // Mesaj gönderme fonksiyonu
    function sendMessage() {
        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        
        if (message !== '') {
            // Kullanıcı mesajını ekle
            const userMessage = createUserMessage(message);
            chatMessages.appendChild(userMessage);
            
            // Sunucuya mesaj gönder
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ağ yanıtı uygun değil');
                }
                return response.json();
            })
            .then(data => {
                if (data.response) {
                    const botMessage = createBotMessage(data.response);
                    chatMessages.appendChild(botMessage);
                } else {
                    const errorMessage = createBotMessage(data.error || 'Bir hata oluştu.');
                    chatMessages.appendChild(errorMessage);
                }
                chatMessages.scrollTop = chatMessages.scrollHeight;
            })
            .catch(error => {
                const errorMessage = createBotMessage('Sunucu hatası: ' + error.message);
                chatMessages.appendChild(errorMessage);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
            
            // Input'u temizle
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }
    }

    // Tema Yönetimi
    function initializeTheme() {
        const currentTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', currentTheme);
    }

    // Event Listeners
    messageInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton?.addEventListener('click', sendMessage);

    menuToggle?.addEventListener('click', () => {
        sidebar?.classList.toggle('visible');
    });

    themeToggle?.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // Sidebar dışı tıklama
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768) {
            if (sidebar && !sidebar.contains(e.target) && menuToggle && !menuToggle.contains(e.target)) {
                sidebar.classList.remove('active');
            }
        }
    });

    // Alt menü yönetimi
    submenuTriggers.forEach(trigger => {
        const submenu = trigger.querySelector('.nav-submenu');
        if (submenu) {
            trigger.addEventListener('mouseenter', () => {
                submenu.style.display = 'flex';
            });
            trigger.addEventListener('mouseleave', () => {
                submenu.style.display = 'none';
            });
        }
    });

    // Chat arama
    searchInput?.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        
        chatItems.forEach(item => {
            const title = item.querySelector('.chat-title')?.textContent.toLowerCase();
            const preview = item.querySelector('.chat-preview')?.textContent.toLowerCase();
            
            if (title?.includes(searchTerm) || preview?.includes(searchTerm)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });

    // Textarea otomatik yükseklik ayarı
    function adjustTextareaHeight(immediate = false) {
        if (!messageInput) return;
        
        // Geçici bir div oluştur ve stili kopyala
        const textareaStyles = window.getComputedStyle(messageInput);
        const hiddenDiv = document.createElement('div');
        
        // Stil kopyalama
        hiddenDiv.style.cssText = `
            visibility: hidden;
            position: absolute;
            width: ${messageInput.clientWidth}px;
            font-size: ${textareaStyles.fontSize};
            font-family: ${textareaStyles.fontFamily};
            line-height: ${textareaStyles.lineHeight};
            padding: ${textareaStyles.padding};
            border: ${textareaStyles.border};
            box-sizing: border-box;
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
        `;
        
        // Div'i sayfaya ekle
        document.body.appendChild(hiddenDiv);
        
        // İçeriği kopyala ve yüksekliği hesapla
        const content = messageInput.value || '.';
        hiddenDiv.textContent = content;
        const requiredHeight = hiddenDiv.clientHeight;
        
        // Div'i kaldır
        document.body.removeChild(hiddenDiv);
        
        // Yüksekliği ayarla
        const setHeight = () => {
            const newHeight = Math.min(Math.max(24, requiredHeight), 200);
            messageInput.style.height = `${newHeight}px`;
            
            // Container yüksekliğini ayarla
            const container = messageInput.closest('.message-input-container');
            if (container) {
                container.style.height = `${newHeight + 20}px`; // padding için +20px
            }
        };
        
        immediate ? setHeight() : requestAnimationFrame(setHeight);
    }

    // Input değiştiğinde yüksekliği ayarla
    messageInput?.addEventListener('input', function() {
        const text = this.value;
        const lines = text.split('\n');
        
        // Her 60 karakterde satır sonu ekle
        const formattedLines = lines.map(line => {
            if (line.length <= 60) return line;
            
            const chunks = [];
            let currentIndex = 0;
            
            while (currentIndex < line.length) {
                chunks.push(line.substr(currentIndex, 60));
                currentIndex += 60;
            }
            
            return chunks.join('\n');
        });
        
        const formattedText = formattedLines.join('\n');
        
        if (text !== formattedText) {
            const start = this.selectionStart;
            const end = this.selectionEnd;
            const addedLineBreaks = (formattedText.match(/\n/g) || []).length - 
                                   (text.match(/\n/g) || []).length;
            
            this.value = formattedText;
            
            this.selectionStart = start + addedLineBreaks;
            this.selectionEnd = end + addedLineBreaks;
        }
        
        adjustTextareaHeight(false);
    });

    // Silme işlemleri için
    messageInput?.addEventListener('keydown', function(e) {
        if (e.key === 'Backspace' || e.key === 'Delete') {
            requestAnimationFrame(() => adjustTextareaHeight(true));
        }
    });

    // Kesme/Yapıştırma işlemleri için
    messageInput?.addEventListener('cut', () => requestAnimationFrame(() => adjustTextareaHeight(true)));
    messageInput?.addEventListener('paste', () => requestAnimationFrame(() => adjustTextareaHeight(false)));

    // Pencere yeniden boyutlandığında
    window.addEventListener('resize', () => {
        if (messageInput) {
            requestAnimationFrame(() => adjustTextareaHeight(true));
        }
    });

    // Sayfa kapatıldığında interval'i temizle
    window.addEventListener('unload', () => {
        if (heightCheckInterval) clearInterval(heightCheckInterval);
    });

    // ESC tuşu ile sidebar kapatma
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && window.innerWidth <= 768) {
            sidebar?.classList.remove('active');
        }
    });

    // Pencere boyutu değişimi
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            sidebar?.classList.remove('active');
            submenuTriggers.forEach(trigger => {
                const submenu = trigger.nextElementSibling;
                if (submenu) submenu.style.display = '';
            });
        }
    });

    // Başlangıç işlemleri
    initializeTheme();
    updateDates();

    // Dosya yükleme işlemleri
    const fileUploadButton = document.getElementById('fileUploadButton');
    const fileInput = document.getElementById('fileInput');

    fileUploadButton?.addEventListener('click', () => {
        fileInput?.click();
    });

    fileInput?.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            // Dosya seçildi, burada dosya işleme mantığınızı ekleyebilirsiniz
            console.log('Seçilen dosyalar:', files);
            
            // Dosya seçildikten sonra input'u temizle
            fileInput.value = '';
        }
    });
});

