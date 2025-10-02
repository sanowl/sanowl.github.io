document.addEventListener('DOMContentLoaded', () => {
  initToggles();
  initQLoRAMemory();
  initPPOClipping();
  initGRPOWeights();
  initFlashAttnEstimator();
  initPanZoom();
  initTilt3D();
  initDragNodes();
  initPrettySliders();
});

function initToggles() {
  document.querySelectorAll('[data-toggle]')?.forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-toggle');
      const el = document.getElementById(id);
      if (el) {
        const open = el.getAttribute('data-open') === 'true';
        el.style.display = open ? 'none' : '';
        el.setAttribute('data-open', (!open).toString());
        btn.textContent = open ? 'Show more' : 'Show less';
      }
    });
  });
}

function byId(id) { return document.getElementById(id); }
function num(id, def=0) {
  const v = parseFloat(byId(id)?.value || def);
  return isFinite(v) ? v : def;
}

// QLoRA memory calculator
function initQLoRAMemory() {
  const root = byId('qlora-calc');
  if (!root) return;
  const update = () => {
    const P_B = num('qc_params_b', 7); // billions
    const P = P_B * 1e9;
    const base_bpp = num('qc_base_bpp', 0.6); // bytes/param
    const scales_bpp = num('qc_scales_bpp', 0.1);
    const cov = num('qc_cov', 0.7);
    const r = num('qc_r', 8);
    const d = num('qc_d', 4096);
    const kv = num('qc_kv_gb', 6);

    const base_gb = (P * base_bpp) / 1e9;
    const scales_gb = (P * scales_bpp) / 1e9;
    const adapter_params = P * cov * (2 * r / d);
    const adapters_gb = (adapter_params * 2) / 1e9; // FP16
    const opt_cpu_gb = adapters_gb * 2; // Adam m,v (rough)
    const total_gb = base_gb + scales_gb + adapters_gb + kv;

    byId('qout_base').textContent = base_gb.toFixed(2) + ' GB';
    byId('qout_scales').textContent = scales_gb.toFixed(2) + ' GB';
    byId('qout_adapters').textContent = adapters_gb.toFixed(2) + ' GB';
    byId('qout_opt').textContent = opt_cpu_gb.toFixed(2) + ' GB (CPU)';
    byId('qout_kv').textContent = kv.toFixed(2) + ' GB';
    byId('qout_total').textContent = total_gb.toFixed(2) + ' GB';

    // simple bar visualization
    const bars = [
      {id:'qbar_base', v:base_gb},
      {id:'qbar_scales', v:scales_gb},
      {id:'qbar_adapters', v:adapters_gb},
      {id:'qbar_kv', v:kv}
    ];
    const maxv = Math.max(...bars.map(b=>b.v), 0.1);
    bars.forEach(b=>{
      const el = byId(b.id);
      if (el) el.style.width = (b.v / maxv * 100).toFixed(1) + '%';
    });
  };
  root.querySelectorAll('input')?.forEach(i=> i.addEventListener('input', update));
  update();
}

// PPO clipping explorer
function initPPOClipping() {
  const root = byId('ppo-clip');
  if (!root) return;
  const update = () => {
    const r = num('ppo_r', 1.0);
    const eps = num('ppo_eps', 0.2);
    const A = parseFloat(document.querySelector('input[name="ppo_A"]:checked')?.value || '1');
    const unclipped = r * A;
    const rc = Math.max(1 - eps, Math.min(1 + eps, r));
    const clipped = rc * A;
    byId('ppo_out_r').textContent = r.toFixed(3);
    byId('ppo_out_unclip').textContent = unclipped.toFixed(3);
    byId('ppo_out_clip').textContent = clipped.toFixed(3);
  };
  root.querySelectorAll('input')?.forEach(i=> i.addEventListener('input', update));
  update();
}

// GRPO listwise weights
function initGRPOWeights() {
  const root = byId('grpo-weights');
  if (!root) return;
  const inputs = ['g_s1','g_s2','g_s3','g_s4'];
  const update = () => {
    const tau = num('g_tau', 1.0);
    const S = inputs.map(id => num(id, 0));
    const e = S.map(s => Math.exp(s / Math.max(1e-6, tau)));
    const Z = e.reduce((a,b)=>a+b,0) || 1;
    const w = e.map(x => x/Z);
    w.forEach((wi, idx) => {
      byId('g_w'+(idx+1)).textContent = wi.toFixed(3);
      const bar = byId('g_bar'+(idx+1));
      if (bar) bar.style.height = (wi*100).toFixed(1)+'%';
    });
  };
  root.querySelectorAll('input')?.forEach(i=> i.addEventListener('input', update));
  update();
}

// FlashAttention simple estimator
function initFlashAttnEstimator() {
  const root = byId('fa-est');
  if (!root) return;
  const update = () => {
    const n = Math.max(1, Math.floor(num('fa_n', 2048)));
    const dq = Math.max(1, Math.floor(num('fa_tq', 128)));
    const dk = Math.max(1, Math.floor(num('fa_tk', 128)));
    const tiles = Math.ceil(n/dq) * Math.ceil(n/dk);
    const naive_scores = n * n; // relative units
    const flash_units = tiles * (dq * dk); // processed in tiles
    const workspace_saved = 100; // avoids n^2 scores
    byId('fa_tiles').textContent = tiles.toString();
    byId('fa_naive').textContent = (naive_scores/1e6).toFixed(2)+'M entries';
    byId('fa_flash').textContent = (flash_units/1e6).toFixed(2)+'M tile‑ops';
    byId('fa_ws').textContent = workspace_saved.toFixed(0)+'%';
  };
  root.querySelectorAll('input')?.forEach(i=> i.addEventListener('input', update));
  update();
}

// Beautify range sliders with filled track percentage
function initPrettySliders() {
  const apply = (sl) => {
    const min = parseFloat(sl.min || '0');
    const max = parseFloat(sl.max || '100');
    const val = parseFloat(sl.value || min);
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    sl.style.setProperty('--val', pct + '%');
  };
  document.querySelectorAll('.controls input[type="range"]').forEach(sl => {
    sl.addEventListener('input', () => apply(sl));
    apply(sl);
  });
}

// Generic pan+zoom for SVGs marked data-panzoom
function initPanZoom() {
  const svgs = document.querySelectorAll('svg[data-panzoom]');
  svgs.forEach(svg => {
    const container = svg.parentElement;
    let state = {scale: 1, tx: 0, ty: 0, panning: false, lx:0, ly:0};
    // Wrap children in a group for transforms
    let g = svg.querySelector('g.pz-root');
    if (!g) {
      g = document.createElementNS('http://www.w3.org/2000/svg','g');
      g.setAttribute('class','pz-root');
      while (svg.firstChild) g.appendChild(svg.firstChild);
      svg.appendChild(g);
    }
    const apply = () => g.setAttribute('transform', `translate(${state.tx},${state.ty}) scale(${state.scale})`);
    const point = (evt) => {
      const rect = svg.getBoundingClientRect();
      return {x: evt.clientX - rect.left, y: evt.clientY - rect.top};
    };
    svg.addEventListener('wheel', (e) => {
      e.preventDefault();
      const dir = e.deltaY > 0 ? -1 : 1;
      const factor = 1 + dir * 0.1;
      const p = point(e);
      const pre = {x:(p.x - state.tx)/state.scale, y:(p.y - state.ty)/state.scale};
      state.scale = Math.max(0.4, Math.min(5, state.scale * factor));
      state.tx = p.x - pre.x * state.scale;
      state.ty = p.y - pre.y * state.scale;
      apply();
    }, {passive:false});
    svg.addEventListener('mousedown', (e) => {
      if ((e.target).closest('[data-draggable]')) return; // let node drag handle it
      state.panning = true; svg.classList.add('panning');
      state.lx = e.clientX; state.ly = e.clientY;
    });
    window.addEventListener('mousemove', (e) => {
      if (!state.panning) return;
      state.tx += (e.clientX - state.lx);
      state.ty += (e.clientY - state.ly);
      state.lx = e.clientX; state.ly = e.clientY;
      apply();
    });
    window.addEventListener('mouseup', () => { state.panning = false; svg.classList.remove('panning'); });
    apply();

    // Controls overlay
    if (container && !container.querySelector('.pz-controls')) {
      container.style.position = 'relative';
      const ctr = document.createElement('div');
      ctr.className = 'pz-controls';
      const bIn = document.createElement('button'); bIn.textContent = '+';
      const bOut = document.createElement('button'); bOut.textContent = '−';
      const bReset = document.createElement('button'); bReset.textContent = 'Reset';
      const bTilt = document.createElement('button'); bTilt.textContent = 'Tilt: off';
      bIn.addEventListener('click', ()=>{ state.scale = Math.min(5, state.scale*1.1); apply(); });
      bOut.addEventListener('click', ()=>{ state.scale = Math.max(0.4, state.scale/1.1); apply(); });
      bReset.addEventListener('click', ()=>{ state.scale=1; state.tx=0; state.ty=0; apply(); });
      bTilt.addEventListener('click', ()=>{
        const active = container.getAttribute('data-tilt-active') === 'true';
        container.setAttribute('data-tilt-active', (!active).toString());
        bTilt.textContent = 'Tilt: ' + (!active ? 'on' : 'off');
      });
      container.setAttribute('data-tilt-active','false');
      ctr.append(bIn,bOut,bReset,bTilt);
      container.appendChild(ctr);
    }
  });
}

// Tilt effect for containers with data-tilt
function initTilt3D() {
  const boxes = document.querySelectorAll('[data-tilt]');
  boxes.forEach(box => {
    const svg = box.querySelector('svg');
    if (!svg) return;
    box.style.perspective = '800px';
    svg.style.transformStyle = 'preserve-3d';
    box.addEventListener('mousemove', (e) => {
      if (box.getAttribute('data-tilt-active') !== 'true') return;
      const r = box.getBoundingClientRect();
      const cx = r.left + r.width/2, cy = r.top + r.height/2;
      const dx = (e.clientX - cx)/r.width; // -0.5..0.5
      const dy = (e.clientY - cy)/r.height;
      const rotX = (+10 * dy).toFixed(2);
      const rotY = (-10 * dx).toFixed(2);
      svg.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
    });
    box.addEventListener('mouseleave', ()=> { svg.style.transform = 'rotateX(0deg) rotateY(0deg)'; });
  });
}

// Drag individual nodes with data-draggable in SVGs
function initDragNodes() {
  const svgs = document.querySelectorAll('svg[data-panzoom]');
  svgs.forEach(svg => {
    let cur = null; let start = {x:0,y:0}; let acc = {dx:0,dy:0};
    const onDown = (e) => {
      const t = e.target.closest('[data-draggable]');
      if (!t) return;
      e.stopPropagation();
      cur = t; start.x = e.clientX; start.y = e.clientY;
      const tr = cur.getAttribute('data-dt') || '0,0';
      const [dx,dy] = tr.split(',').map(parseFloat); acc.dx = dx||0; acc.dy=dy||0;
      cur.style.cursor = 'grabbing';
    };
    const onMove = (e) => {
      if (!cur) return;
      const dx = e.clientX - start.x; const dy = e.clientY - start.y;
      const nx = acc.dx + dx; const ny = acc.dy + dy;
      cur.setAttribute('transform', `translate(${nx},${ny})`);
      cur.setAttribute('data-dt', `${nx},${ny}`);
    };
    const onUp = () => { if (cur) { cur.style.cursor='move'; } cur=null; };
    svg.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  });
}
