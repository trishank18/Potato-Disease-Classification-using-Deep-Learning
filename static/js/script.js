(() => {
  const input = document.getElementById('image-input');
  const preview = document.getElementById('image-preview');
  const dropzone = document.getElementById('dropzone');
  const form = document.getElementById('upload-form');
  const loading = document.getElementById('loading');
  if (!input) return;
  const showPreview = file => {
    if (!file) return;
    if (file.size > 8 * 1024 * 1024) { alert('Please choose an image smaller than 8 MB.'); input.value = ''; return; }
    if (!['image/png', 'image/jpeg'].includes(file.type)) { alert('Please choose a PNG, JPG, or JPEG image.'); input.value = ''; return; }
    preview.src = URL.createObjectURL(file); preview.hidden = false; dropzone.classList.add('has-image');
  };
  input.addEventListener('change', () => showPreview(input.files[0]));
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
  dropzone.addEventListener('drop', event => { const file = event.dataTransfer.files[0]; if (file) { input.files = event.dataTransfer.files; showPreview(file); } });
  form.addEventListener('submit', () => { loading.hidden = false; form.querySelector('button[type="submit"]').disabled = true; });
})();
