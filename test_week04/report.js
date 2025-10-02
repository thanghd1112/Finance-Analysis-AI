document.addEventListener('DOMContentLoaded', function() {
    // Tab chuyển đổi chức năng
    const tabBtns = document.querySelectorAll('.tab-btn');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Loại bỏ active class từ tất cả các buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            // Thêm active class vào button hiện tại
            this.classList.add('active');
            
            // Logic để hiển thị nội dung tab (sẽ cần kết hợp với backend)
            const tabName = this.getAttribute('data-tab');
            console.log('Tab đã chọn:', tabName);
            
            // Đây là nơi bạn sẽ thêm code để thay đổi nội dung hiển thị
            
        });
    });
});