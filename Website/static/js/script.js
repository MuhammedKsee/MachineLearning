// DOM yüklendiğinde çalışacak kodlar
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elementleri
    const sidebar = document.querySelector('.sidebar');
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelectorAll('.nav-link');
    const submenuTriggers = document.querySelectorAll('.nav-link[data-submenu]');
    const searchInput = document.querySelector('.search-input');
    const chatItems = document.querySelectorAll('.chat-item');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');
    const themeToggle = document.getElementById('themeToggle');

    // Tema Yönetimi
    function initializeTheme() {
        const currentTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', currentTheme);
    }

    themeToggle?.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });

    // Sidebar Yönetimi
    menuToggle?.addEventListener('click', () => {
        sidebar.classList.toggle('active');
    });

    // Dışarı tıklama ile sidebar'ı kapat (mobil)
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768) {
            if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                sidebar.classList.remove('active');
            }
        }
    });

    // Alt menüler için hover/click yönetimi
    submenuTriggers.forEach(trigger => {
        const submenu = trigger.nextElementSibling;
        
        if (window.innerWidth <= 768) {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                submenu.style.display = submenu.style.display === 'block' ? 'none' : 'block';
            });
        }
    });

    // Chat Geçmişi Arama
    searchInput?.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        
        chatItems.forEach(item => {
            const title = item.querySelector('.chat-title').textContent.toLowerCase();
            const preview = item.querySelector('.chat-preview').textContent.toLowerCase();
            
            if (title.includes(searchTerm) || preview.includes(searchTerm)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });

    // Chat Item Tıklama
    chatItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Aktif sınıfını kaldır
            chatItems.forEach(chat => chat.classList.remove('active'));
            
            // Seçilen öğeyi aktif yap
            item.classList.add('active');

            // Mobilde sidebar'ı kapat
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
            }
        });
    });

    // Mesaj Gönderme
    function sendMessage() {
        const message = messageInput.value.trim();
        if (message) {
            // Mesaj gönderme işlemleri buraya gelecek
            console.log('Mesaj gönderildi:', message);
            messageInput.value = '';
            adjustTextareaHeight();
        }
    }

    sendButton?.addEventListener('click', sendMessage);

    messageInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Textarea yükseklik ayarı
    function adjustTextareaHeight() {
        if (!messageInput) return;
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    }

    messageInput?.addEventListener('input', adjustTextareaHeight);

    // ESC tuşu ile sidebar'ı kapat (mobil)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && window.innerWidth <= 768) {
            sidebar.classList.remove('active');
        }
    });

    // Pencere boyutu değişimi
    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) {
            sidebar.classList.remove('active');
            submenuTriggers.forEach(trigger => {
                trigger.nextElementSibling.style.display = '';
            });
        }
    });

    // Başlangıç işlemleri
    initializeTheme();
});
