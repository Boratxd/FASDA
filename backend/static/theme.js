function _fasdaToggleTheme() {
    var html = document.documentElement;
    var isLight = html.getAttribute('data-theme') === 'light';
    if (isLight) {
        html.removeAttribute('data-theme');
        localStorage.setItem('fasda-theme', 'dark');
    } else {
        html.setAttribute('data-theme', 'light');
        localStorage.setItem('fasda-theme', 'light');
    }
    _fasdaUpdateToggleBtn();
}

function _fasdaUpdateToggleBtn() {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var isLight = document.documentElement.getAttribute('data-theme') === 'light';
    btn.textContent = isLight ? '🌙' : '☀️';
    btn.title = isLight ? 'Switch to Dark Mode' : 'Switch to Light Mode';
}

document.addEventListener('DOMContentLoaded', function() {
    _fasdaUpdateToggleBtn();

    
    var path = window.location.pathname;
    document.querySelectorAll('.sb-link').forEach(function(link) {
        var href = link.getAttribute('href');
        if (!href) return;
        if (href !== '/' && path.startsWith(href)) {
            link.classList.add('active');
        } else if (href === path) {
            link.classList.add('active');
        }
    });

    
    var hamburger = document.getElementById('hamburger');
    var sidebar = document.querySelector('.sidebar');
    var overlay = document.getElementById('sidebar-overlay');

    if (hamburger && sidebar) {
        hamburger.addEventListener('click', function() {
            sidebar.classList.toggle('open');
            if (overlay) {
                overlay.classList.toggle('active');
                overlay.style.display = overlay.classList.contains('active') ? 'block' : 'none';
            }
        });
    }
    if (overlay) {
        overlay.addEventListener('click', function() {
            if (sidebar) sidebar.classList.remove('open');
            overlay.classList.remove('active');
            overlay.style.display = 'none';
        });
    }
});
