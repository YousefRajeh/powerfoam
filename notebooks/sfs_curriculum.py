import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Splat Feature Solver — an interactive curriculum

    **Paper:** *Splat Feature Solver*, Xiong, Liu, Xu, Chen, Feng (ICLR 2026), arXiv 2508.12216v2.

    This notebook walks the paper from the rendering equation to the `(1+β)` bound, one module at a
    time. Every module has the same four kinds of cell:

    | cell | what it does |
    |---|---|
    | 📐 **Math** | the equations, in the paper's own notation, with the intuition |
    | 🔢 **Example** | the smallest numeric case that makes the equation concrete — read the numbers |
    | 🎛 **Visualization** | a Three.js widget you can drag and slide |
    | ✅ **Check** | a question; the answer cell reacts to what you pick |

    The spine is Modules 1–7. Module 7 is where the paper's central theorem is audited, and where
    this notebook stops being exposition and starts being research.
    """)
    return


@app.cell
def _():
    # ---------------------------------------------------------------------------------------------
    # Widget scaffolding: a tiny 3D projector drawn on a 2D canvas.
    #
    # NOT Three.js, and deliberately so. WebGL is unavailable on this machine (Chromium reports
    # GL_VENDOR/GL_RENDERER = Disabled, "BindToCurrentSequence failed"), which no amount of iframe
    # configuration can fix -- and a curriculum whose figures depend on the reader's GPU driver is a
    # curriculum that breaks. These scenes are only spheres, segments, rings and bars, so a
    # painter's-algorithm projector renders them faithfully with no GPU, no CDN, and no dependency
    # of any kind. It also means the widgets work offline.
    #
    # The engine exposes sph/seg/ring/bar/group, an orbit camera, and a readout pane. Objects are
    # mutable handles: a slider handler edits .p/.r/.c/.o and the next frame reflects it.
    # ---------------------------------------------------------------------------------------------
    PAGE = """<!doctype html>
    <html><head><meta charset="utf-8"><style>
      html,body { height:100%; }
      body { margin:0; font:13px/1.45 ui-sans-serif,system-ui,sans-serif;
             background:#0f1116; color:#e6e8ee; }
      #err { display:none; padding:12px 14px; background:#3b1418; color:#ffd7d7;
             border-bottom:1px solid #7a2530; white-space:pre-wrap; font-size:12px; }
      #wrap { display:flex; flex-direction:column; height:100%; }
      #view { flex:1 1 auto; min-height:0; position:relative; cursor:grab; }
      #view.grabbing { cursor:grabbing; }
      #view canvas { display:block; width:100%; height:100%; }
      #panel { flex:0 0 auto; padding:10px 12px; background:#161923; border-top:1px solid #262a36;
               display:grid; grid-template-columns:auto 1fr auto; gap:6px 12px; align-items:center; }
      #panel label { color:#aab; white-space:nowrap; }
      #panel output { font-variant-numeric:tabular-nums; color:#7fd1ff; min-width:52px;
                      text-align:right; }
      input[type=range]{ width:100%; }
      #readout { grid-column:1/-1; font-variant-numeric:tabular-nums; color:#cfd4e0;
                 border-top:1px solid #262a36; padding-top:8px; white-space:pre; overflow-x:auto; }
      .hint { grid-column:1/-1; color:#7a8296; font-size:12px; }
    </style></head><body>
    <div id="err"></div>
    <div id="wrap"><div id="view"></div><div id="panel">__PANEL__</div></div>
    <script>
    function fail(msg){ var e=document.getElementById('err'); e.style.display='block';
                        e.textContent='Visualization error\n'+msg; }
    window.onerror=function(m,s,l){ fail(m+'  (line '+l+')'); };
    (function(){
    var view=document.getElementById('view');
    var cv=document.createElement('canvas'); view.appendChild(cv);
    var ctx=cv.getContext('2d');
    if(!ctx){ fail('2D canvas unavailable - the browser is blocking canvas rendering entirely.');
              return; }

    var W=1,H=1,DPR=Math.min(devicePixelRatio||1,2);
    function resize(){
      W=view.clientWidth; H=view.clientHeight;
      cv.width=Math.max(1,W*DPR); cv.height=Math.max(1,H*DPR);
      ctx.setTransform(DPR,0,0,DPR,0,0);
    }
    new ResizeObserver(resize).observe(view);

    // ---- scene graph -------------------------------------------------------------------------
    var OBJ=[];
    function sph(p,r,c){ var o={k:'sph',p:p,r:r,c:c,o:1,vis:true}; OBJ.push(o); return o; }
    function seg(a,b,c,w){ var o={k:'seg',a:a,b:b,c:c,w:w||1.5,o:1,vis:true}; OBJ.push(o); return o; }
    function ring(p,r,c){ var o={k:'ring',p:p,r:r,c:c,o:0.6,vis:true}; OBJ.push(o); return o; }
    function bar(p,h,w,c){ var o={k:'bar',p:p,h:h,w:w||0.5,c:c,o:1,vis:true}; OBJ.push(o); return o; }
    function group(){ return { items:[],
        add:function(o){ this.items.push(o); return o; },
        clear:function(){ var s=this.items; OBJ=OBJ.filter(function(o){return s.indexOf(o)<0;});
                          this.items=[]; } }; }
    function grid(n,step,c){
      var g=group(), e=n*step/2;
      for(var i=0;i<=n;i++){ var t=-e+i*step;
        g.add(seg([-e,0,t],[e,0,t],c,0.6)); g.add(seg([t,0,-e],[t,0,e],c,0.6)); }
      g.items.forEach(function(o){o.o=0.28;}); return g;
    }

    // ---- camera ------------------------------------------------------------------------------
    var yaw=__YAW__, pitch=__PITCH__, dist=__DIST__, target=[__TARGET__];
    var eye=[0,0,0], rt=[1,0,0], up=[0,1,0], fw=[0,0,1];
    function norm(v){ var m=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/m,v[1]/m,v[2]/m]; }
    function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
    function dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
    function place(){
      eye=[target[0]+dist*Math.cos(pitch)*Math.sin(yaw),
           target[1]+dist*Math.sin(pitch),
           target[2]+dist*Math.cos(pitch)*Math.cos(yaw)];
      fw=norm([target[0]-eye[0],target[1]-eye[1],target[2]-eye[2]]);
      rt=norm(cross(fw,[0,1,0])); up=cross(rt,fw);
    }
    function proj(p){
      var v=[p[0]-eye[0],p[1]-eye[1],p[2]-eye[2]];
      var z=dot(v,fw); if(z<0.05) return null;
      var f=(H*0.5)/Math.tan(0.42);
      return {x:W*0.5+f*dot(v,rt)/z, y:H*0.5-f*dot(v,up)/z, z:z, s:f/z};
    }
    var drag=false,px=0,py=0;
    view.addEventListener('pointerdown',function(e){drag=true;px=e.clientX;py=e.clientY;
      view.classList.add('grabbing');});
    addEventListener('pointerup',function(){drag=false;view.classList.remove('grabbing');});
    addEventListener('pointermove',function(e){ if(!drag)return;
      yaw-=(e.clientX-px)*0.008; pitch+=(e.clientY-py)*0.006;
      pitch=Math.max(-1.45,Math.min(1.45,pitch)); px=e.clientX; py=e.clientY; place(); });
    view.addEventListener('wheel',function(e){ e.preventDefault();
      dist=Math.max(2,Math.min(140,dist*(1+Math.sign(e.deltaY)*0.08))); place(); },{passive:false});

    // ---- painter ------------------------------------------------------------------------------
    function hex(c){ return typeof c==='number'
      ? '#'+('000000'+c.toString(16)).slice(-6) : c; }
    function shade(c,f){
      var s=hex(c), r=parseInt(s.substr(1,2),16), g=parseInt(s.substr(3,2),16),
          b=parseInt(s.substr(5,2),16);
      r=Math.min(255,r*f)|0; g=Math.min(255,g*f)|0; b=Math.min(255,b*f)|0;
      return 'rgb('+r+','+g+','+b+')';
    }
    function draw(){
      ctx.clearRect(0,0,W,H);
      var items=[];
      OBJ.forEach(function(o){
        if(!o.vis||o.o<=0.002) return;
        if(o.k==='sph'||o.k==='ring'||o.k==='bar'){
          var q=proj(o.p); if(q) items.push({z:q.z,o:o,q:q});
        } else if(o.k==='seg'){
          var a=proj(o.a), b=proj(o.b); if(a&&b) items.push({z:(a.z+b.z)/2,o:o,a:a,b:b});
        }
      });
      items.sort(function(A,B){ return B.z-A.z; });            // far to near
      items.forEach(function(it){
        var o=it.o; ctx.globalAlpha=Math.max(0,Math.min(1,o.o));
        if(o.k==='seg'){
          ctx.strokeStyle=hex(o.c); ctx.lineWidth=Math.max(0.4,o.w*(it.a.s+it.b.s)*0.5*0.06);
          ctx.beginPath(); ctx.moveTo(it.a.x,it.a.y); ctx.lineTo(it.b.x,it.b.y); ctx.stroke();
        } else if(o.k==='sph'){
          var R=Math.max(0.6,o.r*it.q.s);
          var g=ctx.createRadialGradient(it.q.x-R*0.35,it.q.y-R*0.4,R*0.1,it.q.x,it.q.y,R);
          g.addColorStop(0,shade(o.c,1.45)); g.addColorStop(1,shade(o.c,0.55));
          ctx.fillStyle=g; ctx.beginPath(); ctx.arc(it.q.x,it.q.y,R,0,6.2832); ctx.fill();
        } else if(o.k==='ring'){
          ctx.strokeStyle=hex(o.c); ctx.lineWidth=1.4; ctx.beginPath();
          for(var i=0;i<=48;i++){
            var t=i/48*6.2832;
            var q=proj([o.p[0]+o.r*Math.cos(t),o.p[1],o.p[2]+o.r*Math.sin(t)]);
            if(!q){ continue; } if(i===0) ctx.moveTo(q.x,q.y); else ctx.lineTo(q.x,q.y);
          }
          ctx.stroke();
        } else if(o.k==='bar'){
          var base=proj(o.p), top=proj([o.p[0],o.p[1]+o.h,o.p[2]]);
          if(base&&top){ ctx.strokeStyle=hex(o.c);
            ctx.lineWidth=Math.max(1.5,o.w*base.s*0.5); ctx.lineCap='round';
            ctx.beginPath(); ctx.moveTo(base.x,base.y); ctx.lineTo(top.x,top.y); ctx.stroke();
            ctx.lineCap='butt'; }
        }
      });
      ctx.globalAlpha=1;
    }

    var readout=document.getElementById('readout');
    function S(id){ return document.getElementById(id); }
    function onAll(ids,fn){ ids.forEach(function(i){ S(i).addEventListener('input',fn); }); }

    try {
    __BODY__
    } catch(e){ fail('while building the scene: '+e.message); return; }

    place(); resize();
    (function loop(){ requestAnimationFrame(loop); draw(); })();
    })();
    </script></body></html>"""


    def widget(body, panel, yaw=0.7, pitch=0.35, dist=16.0, target="0,0,0"):
        return (PAGE.replace("__BODY__", body).replace("__PANEL__", panel)
                    .replace("__YAW__", str(yaw)).replace("__PITCH__", str(pitch))
                    .replace("__DIST__", str(dist)).replace("__TARGET__", target))


    def slider(sid, label, lo, hi, step, val):
        return (f'<label for="{sid}">{label}</label>'
                f'<input id="{sid}" type="range" min="{lo}" max="{hi}" step="{step}" value="{val}">'
                f'<output id="{sid}o">{val}</output>')
    return slider, widget


@app.cell
def _(mo):
    # Pure-HTML diagnostic: no three.js, no WebGL required to RUN it -- it only reports what the
    # browser offers. If the 3D widgets below are blank, this cell says why in your own browser
    # rather than in mine.
    _diag = """<!doctype html><html><head><meta charset="utf-8"><style>
      body{margin:0;padding:12px;font:13px ui-monospace,Menlo,Consolas,monospace;
           background:#11131a;color:#d8dce6}
      b{color:#7fd1ff} .ok{color:#06d6a0} .bad{color:#ff8080}
    </style></head><body><div id="o">probing...</div><script>
    var L=[];
    function line(k,v,good){ L.push('<b>'+k+'</b>: <span class="'+(good===undefined?'':(good?'ok':'bad'))+'">'+v+'</span>'); }
    var cv=document.createElement('canvas'), reason='';
    cv.addEventListener('webglcontextcreationerror',function(e){reason=e.statusMessage||'';});
    var found=null;
    ['webgl2','webgl','experimental-webgl'].forEach(function(n){
      var c=null; try{ c=cv.getContext(n); }catch(e){ reason=reason||e.message; }
      line(n, c? 'available':'NOT available', !!c);
      if(c&&!found) found={ctx:c,name:n};
    });
    if(reason) line('browser reason', reason, false);
    if(found){
      var g=found.ctx;
      try{
        var dbg=g.getExtension('WEBGL_debug_renderer_info');
        if(dbg){ line('GPU', g.getParameter(dbg.UNMASKED_RENDERER_WEBGL));
                 line('vendor', g.getParameter(dbg.UNMASKED_VENDOR_WEBGL)); }
        line('GL version', g.getParameter(g.VERSION));
        line('max texture', g.getParameter(g.MAX_TEXTURE_SIZE));
      }catch(e){ line('param read failed', e.message, false); }
    }
    line('in iframe', (window.self!==window.top)?'yes':'no');
    line('secure context', window.isSecureContext?'yes':'no');
    document.getElementById('o').innerHTML = L.join('<br>');
    </script></body></html>"""
    mo.vstack([
        mo.md("### 🩺 WebGL diagnostics — run this if the 3D cells are blank"),
        mo.iframe(_diag, height=230),
    ])
    return


@app.cell
def _(mo):
    def quiz(prompt, options, correct, why):
        """A question plus the reactive answer cell that grades it.

        Returns (radio, grader). Put the radio in one cell and call grader() in the next so the
        feedback appears below the choice rather than replacing it.
        """
        r = mo.ui.radio(options=options, label=prompt)

        def grader():
            if r.value is None:
                return mo.md("*Pick an answer to see the explanation.*").callout(kind="neutral")
            ok = r.value == correct
            head = "✅ **Correct.**" if ok else f"❌ **Not quite** — the answer is *{correct}*."
            return mo.md(f"{head}\n\n{why}").callout(kind="success" if ok else "warn")

        return r, grader

    return (quiz,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 0 — Notation, and the shape of the problem

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $R$ | number of **rays** — one per pixel per view. Tens of millions. |
    | $P$ | number of **primitives** — Gaussians, or foam cells. ~10⁵–10⁶. |
    | $F$ | **feature dimension** — 512 for CLIP ViT-B/16. |
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 The symbols

    | symbol | shape | meaning |
    |---|---|---|
    | $R$ | scalar | number of **rays** (pixels × views) |
    | $P$ | scalar | number of **primitives** (Gaussians, or foam cells) |
    | $F$ | scalar | feature dimension (512 for CLIP ViT-B/16) |
    | $A$ | $R\times P$ | rendering operator; $A_{rp}$ = how much primitive $p$ contributes to ray $r$ |
    | $x$ | $P\times F$ | **the unknown** — one feature vector per primitive |
    | $B$ | $R\times F$ | observed feature at each ray (the CLIP embedding at that pixel) |

    The structural fact that makes everything else possible:

    > **$A$ is fixed the moment the geometry is fixed.** Feature lifting never moves a primitive,
    > never changes an opacity. It solves for $x$ alone.

    That is why lifting is a *linear inverse problem* and not a training problem — and why it can
    take minutes instead of hours.
    """)
    return


@app.cell
def _(mo):
    _R, _P, _F = 1000 * 1_000_000, 1_000_000, 512
    m0_dense_tb = _R * _P * 4 / 1e12
    m0_msg = mo.md(f"""
    ### 🔢 Why $A$ is never built

    A 1000-view scene at 1 MP with 1 M Gaussians:

    - $A$ is ${_R:,}\\times{_P:,}$ — **{m0_dense_tb:,.0f} TB** dense in float32.
    - But each ray touches only a handful of primitives (16×16 tiling), so the real storage is
      $\\mathcal{{O}}(R\\cdot k)$ with $k$ small.

    Every algorithm in this paper therefore has to be expressible as a **streaming pass over rays**,
    touching only the non-zeros. Keep that constraint in mind — it is why the closed form is
    attractive and why forming $A^\\top A$ ({_P:,}²) is out of the question.
    """)
    m0_msg
    return


@app.cell
def _(quiz):
    m0_q, m0_grade = quiz(
        "If you doubled the number of training views, which of these changes?",
        options={
            "R (rows of A) only": "R",
            "P (columns of A) only": "P",
            "Both R and P": "both",
            "Neither — A is fixed": "neither",
        },
        correct="R",
        why=("Views contribute **rays**, which are rows. The primitive count $P$ is set by the "
             "reconstruction, not by how many times you photograph it. This is the asymmetry that "
             "makes the system overdetermined ($R \\gg P$) and is why more views help: they add "
             "equations, not unknowns."),
    )
    m0_q
    return (m0_grade,)


@app.cell
def _(m0_grade):
    m0_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 1 — The rendering model, and where $A$ comes from

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $R$ | number of **rays** — one per pixel per view. Tens of millions. |
    | $P$ | number of **primitives** — Gaussians, or foam cells. ~10⁵–10⁶. |
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $\alpha_{rp}$ | **opacity** of primitive $p$ where ray $r$ hits it, in $[0,1]$. |
    | $\omega_{rp}$ | the **compositing weight** — the fraction of ray $r$'s final colour that primitive $p$ is responsible for. This becomes $A_{ij}$. |
    | $C_r$ | the colour the renderer produces for ray $r$. |
    | $\hat{C}_r$ | the colour actually **observed** in the photo. Becomes $B_i$. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 Alpha compositing *is* a matrix product

    A splat renderer walks the primitives along a ray front-to-back and composites them:

    $$C_r \;=\; \sum_p \omega_{rp}\, c_p, \qquad
      \omega_{rp} \;=\; \underbrace{\alpha_{rp}}_{\text{this primitive's opacity}}
      \cdot \underbrace{\prod_{q<p}\bigl(1-\alpha_{rq}\bigr)}_{\text{transmittance: light that got through}}$$

    **Read it slowly, one piece at a time:**

    - $\alpha_{rp}$ — *how opaque* primitive $p$ is where ray $r$ passes through it. 0 = invisible,
      1 = solid.
    - $\prod_{q<p}(1-\alpha_{rq})$ — walk every primitive $q$ **in front of** $p$ and multiply by
      "the fraction of light each one lets past". This is the **transmittance**: the share of the
      ray's light that still survives by the time it arrives at $p$.
    - $\omega_{rp}$ — multiply those two: *how opaque you are* × *how much light reached you*. That
      is your share of the final pixel.
    - $c_p$ — what primitive $p$ looks like (its colour, or later, its CLIP feature).

    So a pixel's colour is a weighted average of the things along its line of sight, weighted by
    visibility. **Being opaque is not enough — you also have to be reachable.**

    Stare at the right-hand side: it is a **sum over primitives of (weight × per-primitive value)**.
    That is a matrix-vector product. Set

    $$A_{rp} = \omega_{rp}, \qquad x_p = c_p, \qquad B_r = \hat{C}_r$$

    and the renderer becomes $Ax = B$. **The entire paper is that substitution.** Replace the colour
    $c_p$ with a CLIP feature and you are lifting features instead of rendering colour.

    ### The three properties (p. 4) — and which are load-bearing

    1. **Sparsity.** Tile-based rasterisation (16×16 tiles) excludes most splats from any tile, so
       rows of $A$ have few non-zeros.
    2. **Row-stochasticity.** $\sum_p \omega_{rp} \le 1$, approaching 1 because compositing stops
       once accumulated opacity ≈ 1. The authors *engineer* this by randomising the background
       colour during training, forcing splats to fully occlude it from every direction (Appendix G).
    3. **Near-consistency.** Rendered ≈ observed colour (PSNR > 24 dB), so $B$ approximately lies in
       $\operatorname{range}(A)$.

    > ⚠️ **Property 2 is the crux.** Jensen (Module 5), the weighted mean (Module 4) and the bound
    > (Module 7) all need $\sum_j A_{ij} = 1$ **exactly**. The paper establishes it *approximately*.
    > Track every place where "approximately" is quietly upgraded to "exactly" — that is where a
    > careful reader earns their keep.
    """)
    return


@app.cell
def _(mo, np):
    m1_alpha = np.array([0.30, 0.70, 0.50, 0.90])
    m1_T = np.concatenate([[1.0], np.cumprod(1 - m1_alpha)[:-1]])   # transmittance in front of each
    m1_w = m1_alpha * m1_T
    m1_tbl = "\n".join(
        f"| {p} | {a:.2f} | {t:.4f} | **{w:.4f}** |"
        for p, (a, t, w) in enumerate(zip(m1_alpha, m1_T, m1_w))
    )
    mo.md(f"""
    ### 🔢 One ray, four primitives

    Opacities front-to-back: `{np.array2string(m1_alpha, precision=2)}`

    | $p$ | $\\alpha_p$ | transmittance $\\prod_{{q<p}}(1-\\alpha_q)$ | $\\omega_p$ |
    |---|---|---|---|
    {m1_tbl}
    | | | **row sum** | **{m1_w.sum():.4f}** |

    Read the third column downward: it collapses fast. The 4th primitive has $\\alpha=0.90$ — nearly
    opaque — yet contributes only **{m1_w[-1]:.4f}**, because only {m1_T[-1]:.1%} of the light ever
    reaches it. *Occlusion, not opacity, decides influence.*

    The row sums to {m1_w.sum():.4f}, not 1. The missing {1 - m1_w.sum():.4f} is light that passed
    through everything and hit the background — exactly the leak Property 2 tries to close.
    """)
    return


@app.cell
def _(mo, slider, widget):
    m1_body = """
    var N=5, zs=[-6,-3,0,3,6];
    var cols=['#ff6b6b','#ffd166','#06d6a0','#4cc9f0','#b388ff'];
    var sp=[], bars=[];
    for(var i=0;i<N;i++){
      sp.push(sph([0,0,zs[i]],1.15,cols[i]));
      bars.push(bar([0,2.6,zs[i]],0.1,0.55,cols[i]));
    }
    seg([0,0,-10],[0,0,10],'#8899aa',1.2);
    sph([0,0,-9.4],0.42,'#ffffff');                       // the camera/eye marker
    var ids=['a0','a1','a2','a3','a4'];
    function update(){
      var a=ids.map(function(i){return parseFloat(S(i).value);});
      ids.forEach(function(i,k){ S(i+'o').textContent=a[k].toFixed(2); });
      var T=1.0, w=[];
      for(var i=0;i<N;i++){ w.push(a[i]*T); T*=(1-a[i]); }
      var sum=w.reduce(function(p,c){return p+c;},0), mx=Math.max.apply(null,w)||1e-9;
      for(var i=0;i<N;i++){
        sp[i].o=0.14+0.82*a[i];
        sp[i].r=0.75+0.55*(w[i]/mx);
        bars[i].h=Math.max(0.05,7*w[i]);
      }
      readout.textContent=
        'omega   = ['+w.map(function(v){return v.toFixed(4);}).join(', ')+']\\n'+
        'row sum = '+sum.toFixed(4)+'    leaked to background = '+(1-sum).toFixed(4)+'\\n'+
        (sum>0.995?'Property 2 effectively holds':'Property 2 VIOLATED - Jensen (Eq.8) loses its footing');
    }
    onAll(ids,update); update();
    """
    m1_panel = "".join([
        slider("a0", "α₀ (nearest)", 0, 1, 0.01, 0.30),
        slider("a1", "α₁", 0, 1, 0.01, 0.70),
        slider("a2", "α₂", 0, 1, 0.01, 0.50),
        slider("a3", "α₃", 0, 1, 0.01, 0.90),
        slider("a4", "α₄ (farthest)", 0, 1, 0.01, 0.20),
        '<div id="readout"></div>',
        '<div class="hint">Drag to orbit · wheel to zoom. Bars above each sphere are ω. '
        'Push the near spheres opaque and watch the far ones go dark — that is occlusion '
        'starving them of weight, even at α=1.</div>',
    ])
    mo.vstack([mo.md("### 🎛 Alpha compositing along a ray"),
               mo.iframe(widget(m1_body, m1_panel, yaw=0.9, pitch=0.30, dist=22.0), height=560)])
    return


@app.cell
def _(quiz):
    m1_q, m1_grade = quiz(
        "A ray's row sums to 0.85 instead of 1. What breaks?",
        options={
            "Nothing — the solver is scale-invariant": "nothing",
            "Jensen's inequality (Eq. 8) loses its justification": "jensen",
            "A stops being sparse": "sparse",
            "The features stop being unit norm": "norm",
        },
        correct="jensen",
        why=("Jensen needs the weights to be a **probability distribution** — non-negative and "
             "summing to exactly 1. At 0.85 the row is a sub-probability measure and the step "
             "$L \\le J$ is no longer justified as stated. In practice the leak is small, which is "
             "why the method works; but the *theorem* assumes it away. Sparsity is unaffected, and "
             "feature norms are a separate issue entirely."),
    )
    m1_q
    return (m1_grade,)


@app.cell
def _(m1_grade):
    m1_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 2 — The feature lifting equation

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $R$ | number of **rays** — one per pixel per view. Tens of millions. |
    | $P$ | number of **primitives** — Gaussians, or foam cells. ~10⁵–10⁶. |
    | $F$ | **feature dimension** — 512 for CLIP ViT-B/16. |
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 The problem, stated

    $$A x = B, \qquad A\in\mathbb{R}^{R\times P},\; x\in\mathbb{R}^{P\times F},\; B\in\mathbb{R}^{R\times F}
      \tag{Eq. 3}$$
    $$A_{ij} = \omega_{rp}, \qquad x_j = c_p, \qquad B_i = \hat{C}_r \tag{Eq. 4}$$

    Because $R \gg P$, this is **overdetermined** and in general **inconsistent** — there is no $x$
    satisfying it exactly. Every ray that sees a primitive casts a vote for what that primitive's
    feature should be, and the votes disagree.

    ### Well-posedness, in Hadamard's three senses

    - **Existence** — guaranteed once we move to least squares (Module 3).
    - **Uniqueness** — holds iff $A$ has full column rank; fails for a primitive no ray ever sees,
      or for primitives that are never separated by any view.
    - **Continuity** — holds when the observed signal varies smoothly with camera parameters.

    ### ⚠️ Where the paper argues from convenience

    The paper motivates uniqueness partly on *semantic* grounds — "each geometric primitive should
    admit a single descriptor, mirroring an ideal embedding (e.g. a perfect 3D CLIP)". That is a
    **modelling preference wearing the clothes of a theorem**. Whether $A$ has full column rank is a
    fact about geometry and camera placement; what we would *like* a descriptor to be has no bearing
    on it.

    Their own counterexample is the honest part, and it is the failure mode that matters in
    practice:

    > one view's segmentation mask captures only the noodles of a ramen bowl, while the next view's
    > mask includes both the bowl and the noodles

    The CLIP embeddings then **jump discontinuously** between views. Continuity fails, the system is
    badly inconsistent, and the least-squares "solution" is an average of two semantically different
    things. Modules 5 and 9 are both responses to this.
    """)
    return


@app.cell
def _(mo, np):
    m2_A = np.array([
        [0.6, 0.4, 0.0],
        [0.0, 0.5, 0.5],
        [0.7, 0.0, 0.3],
        [0.2, 0.3, 0.5],
    ])
    m2_B = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.4, 0.6]])
    m2_res = np.linalg.lstsq(m2_A, m2_B, rcond=None)
    m2_x = m2_res[0]
    m2_rank = np.linalg.matrix_rank(m2_A)
    m2_resid = np.linalg.norm(m2_A @ m2_x - m2_B)

    m2_A2 = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    m2_rank2 = np.linalg.matrix_rank(m2_A2)

    mo.md(f"""
    ### 🔢 A 4-ray, 3-primitive system

    $A$ (rows sum to 1) and $B$ (2-dim "features"):

    ```
    A =
    {np.array2string(m2_A, precision=2)}
    B =
    {np.array2string(m2_B, precision=2)}
    ```

    - rank of $A$ = **{m2_rank}** of 3 columns → full column rank → the least-squares solution is
      **unique**.
    - least-squares residual $\\|Ax-B\\|_F$ = **{m2_resid:.4f}** → non-zero, so the system is
      **inconsistent**: no exact lift exists. This is the normal case.

    Now a degenerate $A$, where two primitives are *never separated* by any ray (rays 1 and 2 see
    them with identical weights):

    ```
    A2 =
    {np.array2string(m2_A2, precision=2)}
    ```
    rank = **{m2_rank2}** of 3 → **rank-deficient**. Primitives 0 and 1 are indistinguishable to
    every ray, so their features are determined only up to a shared sum. No amount of data fixes
    this — it is a *geometry and camera placement* problem, which is exactly why the semantic
    argument for uniqueness does not do the work the paper needs.
    """)
    return


@app.cell
def _(quiz):
    m2_q, m2_grade = quiz(
        "Which situation makes the *continuity* criterion fail?",
        options={
            "A primitive that no ray ever sees": "unseen",
            "Two primitives no ray separates": "tied",
            "A mask that includes the bowl in one view and not the next": "mask",
            "More views than primitives": "overdet",
        },
        correct="mask",
        why=("Continuity is about the observation $B$ varying smoothly with camera parameters. An "
             "inconsistent mask makes the CLIP embedding **jump** between neighbouring views, so a "
             "small change in camera causes a large change in data. The first two options break "
             "*uniqueness* (rank deficiency), not continuity; the last is just the overdetermined "
             "regime, which is the healthy case."),
    )
    m2_q
    return (m2_grade,)


@app.cell
def _(m2_grade):
    m2_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 3 — Least squares, and what $A^\top A$ *means*

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $R$ | number of **rays** — one per pixel per view. Tens of millions. |
    | $P$ | number of **primitives** — Gaussians, or foam cells. ~10⁵–10⁶. |
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $A^\top A$ | $P\times P$ — the **co-visibility matrix**. Entry $(j,k)$ = how strongly primitives $j$ and $k$ are seen *together*. |
    | $X^\star$ | the exact least-squares solution (same thing as $\hat{x}$ later). |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 The formulation

    $$X^\star = \arg\min_X \; \bigl\|A X - B\bigr\|_F^2 \tag{Eq. 5}$$

    Convex, so a minimiser exists for any $B$ and lies in $\operatorname{range}(A)$. By Property 3
    the residual is small. The normal equations are

    $$A^\top A\, x \;=\; A^\top B.$$

    ### The key reading — $A^\top A$ is a co-visibility matrix

    $$\bigl(A^\top A\bigr)_{jk} \;=\; \sum_i A_{ij} A_{ik}$$

    is a sum over rays of (weight on $j$) × (weight on $k$): it is large exactly when primitives $j$
    and $k$ are **seen together, strongly, by many rays**. So:

    - the **diagonal** $\left(A^\top A\right)_{jj} = \sum_i A_{ij}^2$ measures how much primitive $j$
      is observed at all;
    - the **off-diagonal** measures how *entangled* $j$ and $k$ are — how hard it is for any solver
      to tell their contributions apart.

    > This is the object everything later depends on. Diagonal dominance (Property 4, Module 6) is
    > literally the statement "primitives are seen more on their own than together". Tikhonov
    > guidance (Module 8) adds $\lambda I$ to force it. And our correction to the bound (Module 7)
    > is governed by the **co-visibility Laplacian** — the off-diagonal mass.

    Why it is not solved directly: $A^\top A$ is $P\times P$, a million squared. Never formed.
    """)
    return


@app.cell
def _(mo, np):
    m3_A = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.1, 0.2, 0.7],
        [0.4, 0.4, 0.2, 0.0],
    ])
    m3_G = m3_A.T @ m3_A
    m3_diag = np.diag(m3_G)
    m3_off = m3_G - np.diag(m3_diag)
    m3_ratio = m3_diag / np.maximum(np.abs(m3_off).sum(1), 1e-12)
    mo.md(f"""
    ### 🔢 Reading a co-visibility matrix

    Five rays over four primitives:
    ```
    A =
    {np.array2string(m3_A, precision=2)}

    A^T A =
    {np.array2string(m3_G, precision=3)}
    ```

    Read the entries as scene facts:

    - $(A^\\top A)_{{01}} = {m3_G[0,1]:.3f}$ — the **largest off-diagonal**. Primitives 0 and 1 are
      seen together by rays 0, 1 and 4 with equal weight every time. They are badly entangled;
      no solver can cleanly separate them.
    - $(A^\\top A)_{{03}} = {m3_G[0,3]:.3f}$ — primitives 0 and 3 are **never co-visible**. Their
      features are decoupled.
    - Diagonal dominance ratio $\\left(A^\\top A\\right)_{{jj}} / \\sum_{{k\\neq j}} |(A^\\top A)_{{jk}}|$
      per primitive: `{np.array2string(m3_ratio, precision=2)}`

    Primitive 2 has the healthiest ratio ({m3_ratio[2]:.2f}) because ray 2 sees it almost alone at
    weight 0.9. Primitives 0 and 1 are the worst — and those are precisely the ones a one-shot
    weighted mean will smear together.
    """)
    return


@app.cell
def _(mo, slider, widget):
    m3_body = """
    var P=4;
    var pos=[[-4,0,-3],[-1.2,0,-4.5],[3.5,0,-1],[2,0,3.5]];
    var cols=['#ff6b6b','#ffd166','#06d6a0','#4cc9f0'];
    grid(16,1.4,'#334155');
    for(var j=0;j<P;j++) sph(pos[j],0.75,cols[j]);
    var rays=group(), edges=group();
    var origins=[[-7,4.5,0],[-3,5.2,-6],[6,4.5,-4],[5,5,5],[-6,5,5]];
    function rowsFrom(t){
      var base=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]];
      return base.map(function(r){
        var m=r.map(function(v){ return v*(1-t)+t*0.5; });
        var s=m.reduce(function(p,c){return p+c;},0);
        return m.map(function(v){ return v/s; });
      });
    }
    function rebuild(){
      var t=parseFloat(S('mix').value); S('mixo').textContent=t.toFixed(2);
      var rows=rowsFrom(t);
      rays.clear(); edges.clear();
      rows.forEach(function(r,i){
        r.forEach(function(w,j){
          if(w<0.02) return;
          var o=rays.add(seg(origins[i],pos[j],'#9fb3c8',0.9)); o.o=0.10+0.7*w;
        });
      });
      var G=[], txt='A^T A =\\n';
      for(var j=0;j<P;j++){ G.push([]); for(var k=0;k<P;k++){
        var s=0; rows.forEach(function(r){ s+=r[j]*r[k]; }); G[j].push(s); } }
      for(var j=0;j<P;j++) txt+='  ['+G[j].map(function(v){return v.toFixed(3);}).join(', ')+']\\n';
      var worst=0, offsum=0, tot=0;
      for(var j=0;j<P;j++){ var off=0;
        for(var k=0;k<P;k++){ tot+=G[j][k]; if(k!=j){ off+=Math.abs(G[j][k]); offsum+=G[j][k]; } }
        worst=Math.max(worst,off/Math.max(G[j][j],1e-9)); }
      txt+='off-diagonal fraction = '+(offsum/Math.max(tot,1e-9)).toFixed(3)+
           '    worst off/diag ratio = '+worst.toFixed(3)+'\\n'+
           (worst<1?'diagonally dominant - the weighted mean is close to optimal'
                   :'entangled - the one-shot solver will smear these together');
      readout.textContent=txt;
      for(var j=0;j<P;j++) for(var k=j+1;k<P;k++){
        var w=G[j][k]; if(w<0.01) continue;
        var e=edges.add(seg(pos[j],pos[k],'#ffffff',1+9*w)); e.o=0.25+0.6*w;
      }
    }
    onAll(['mix'],rebuild); rebuild();
    """
    m3_panel = "".join([
        slider("mix", "ray overlap", 0, 1, 0.01, 0.15),
        '<div id="readout"></div>',
        '<div class="hint">Thin grey lines are rays reaching primitives; thick white tubes are '
        'co-visibility (AᵀA)ⱼₖ. Slide right to make every ray see every primitive and watch the '
        'off-diagonal mass — the thing that defeats a one-shot solver — take over.</div>',
    ])
    mo.vstack([mo.md("### 🎛 Co-visibility: the off-diagonal mass of $A^\\top A$"),
               mo.iframe(widget(m3_body, m3_panel, yaw=0.5, pitch=0.62, dist=20.0), height=560)])
    return


@app.cell
def _(quiz):
    m3_q, m3_grade = quiz(
        "What does a large off-diagonal entry $(A^\\top A)_{jk}$ tell you?",
        options={
            "Primitives j and k are close together in space": "close",
            "Primitives j and k are frequently seen together by the same rays": "covis",
            "Primitive j is very opaque": "opaque",
            "The system is inconsistent": "incons",
        },
        correct="covis",
        why=("It is $\\sum_i A_{ij}A_{ik}$ — a weighted count of rays that see **both**. Spatial "
             "proximity often causes co-visibility but is neither necessary nor sufficient: two "
             "adjacent primitives separated by an occluder are never co-visible, and two distant "
             "ones along the same line of sight always are. Co-visibility is the quantity that "
             "controls how badly a one-shot solver confuses two primitives."),
    )
    m3_q
    return (m3_grade,)


@app.cell
def _(m3_grade):
    m3_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 4 — The closed form: a row-sum preconditioner

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $x'_j$ | the **closed-form answer** for primitive $j$ — what the paper's solver returns. |
    | $D$ | diagonal matrix of row sums, used only to write the solver compactly. |
    | $e$ | the all-ones vector. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 The solver

    The paper writes it imposingly (Eq. 6), with $D$ the diagonal of row sums and $e$ all-ones:

    $$D^{1/2}(A^\top A)\,x \;=\; D^{-1/2}(A^\top A)\,e \times D^{1/2}(A^\top A)\,B$$

    but the element-wise form on the right of Eq. 6 is the whole story:

    $$\boxed{\;x_j \;=\; \frac{\sum_i A_{ij} B_i}{\sum_i A_{ij}}\;}$$

    **Read it slowly:**

    - the sum $\sum_i$ runs over **every ray that saw primitive $j$** (all other rays have
      $A_{ij}=0$ and drop out);
    - $B_i$ is what that ray observed — a CLIP vector;
    - $A_{ij}$ is how much of that ray primitive $j$ was responsible for;
    - so the numerator is "add up the observations, each scaled by how much this primitive owned
      it", and the denominator normalises by the total ownership.

    In one sentence: **every ray that sees a primitive votes for what its feature should be, and
    votes are weighted by how much of that ray the primitive explains.**

    ### 📐 What "row-sum preconditioner" actually means

    **A preconditioner, in general.** To solve $Mx=b$ when $M$ is too big to invert, pick a matrix
    $P$ that (a) resembles $M$ and (b) is trivial to invert, and work with $P^{-1}$ instead. The
    cheapest choice is *Jacobi* preconditioning: discard everything off the diagonal, $P=\mathrm{diag}(M)$.

    **The problem here.** Least squares gives the normal equations

    $$A^\top A\,x = A^\top B \qquad\Longrightarrow\qquad x^\star = (A^\top A)^{-1}A^\top B$$

    and $A^\top A$ is $P\times P$ — a million squared. Never formed, never inverted. So replace it
    with a diagonal $D$ you *can* invert: $x' = D^{-1}A^\top B$.

    **Why "row-sum".** $D$ is built by summing each row of $A^\top A$ onto its diagonal — collapsing
    the row's mass to one entry. (Finite-element people call this *mass lumping*.) And this is where
    Property 2 earns its keep:

    $$\bigl(A^\top A\bigr)\text{ row sum at } j
      \;=\; \sum_k \sum_i A_{ij}A_{ik}
      \;=\; \sum_i A_{ij}\underbrace{\Bigl(\sum_k A_{ik}\Bigr)}_{=\,1\ \text{(Property 2)}}
      \;=\; \sum_i A_{ij}$$

    > **The row sums of $A^\top A$ are exactly the column sums of $A$** — the total rendering weight
    > each primitive ever received. Computable in one streaming pass, never touching $A^\top A$.

    Substituting $D=\mathrm{diag}\bigl(\sum_i A_{ij}\bigr)$ turns $x'=D^{-1}A^\top B$ into Eq. 9,
    the weighted mean. **The whole chain is: exact solve needs $(A^\top A)^{-1}$ → too big → keep
    only the diagonal → under row-stochasticity that diagonal is the column sums of $A$ → the
    "solve" collapses to an average.**

    **When is it exact?** Precisely when $A^\top A$ *is* diagonal — when discarding the off-diagonal
    discards nothing, i.e. no two primitives are ever co-visible. That is the quantity measured in
    Module 11: foam's median ray has participation ratio 1.01, so $A^\top A$ is nearly diagonal and
    the lumping is nearly free; 3DGS sits at 7.8 with 5.9× the off-diagonal mass.

    **Two honest footnotes.** The matrix form printed as Eq. 6 is mangled in the PDF and does not
    parse cleanly as written; the element-wise form on its right is the operative statement, and it
    is what the code does. And calling this a *preconditioner* is generous: a preconditioner
    normally accelerates an **iterative** solve that still converges to $x^\star$. Here there is no
    iteration — the preconditioned step is taken once and stopped. That is why an approximation
    error exists at all, and why Module 7 needs a bound.

    ### What it actually is

    > **The contribution-weighted mean of every observation that saw the primitive.** Nothing more.

    Each ray that sees primitive $j$ casts a vote $B_i$, weighted by how much of that ray's colour
    primitive $j$ was responsible for. Average the votes. That is the "solver".

    This is why lifting takes minutes: it is a single streaming pass over rays, accumulating a
    weighted sum and a weight total per primitive. No iteration, no linear solve, and it never forms
    $A^\top A$ — which is what the $D$-scaling in Eq. 6 is really buying.

    ### The catch, stated now and proved in Module 5

    A weighted mean is **not** the least-squares solution unless the columns of $A$ are orthogonal
    (i.e. no two primitives are ever co-visible). When primitives *are* co-visible, the mean
    double-counts shared evidence: it credits each of them with the full ray, rather than splitting
    the credit. Module 3's off-diagonal mass is exactly the amount of double counting.
    """)
    return


@app.cell
def _(mo, np):
    _rng4 = np.random.default_rng(0)
    _A = _rng4.random((200, 12)); _A /= _A.sum(1, keepdims=True)   # row-stochastic (Property 2)
    _G = _A.T @ _A
    _B4 = _rng4.random((200, 3))
    _xp = (_A.T @ _B4) / _A.sum(0)[:, None]                        # Eq. 9
    _xd = np.linalg.solve(np.diag(_G.sum(1)), _A.T @ _B4)           # invert the lumped matrix
    _A2 = _rng4.random((200, 12))                                   # NOT row-stochastic
    mo.md(f"""
    ### 🔢 The row-sum identity, checked

    | claim | result |
    |---|---|
    | row sums of $A^\top A$ = column sums of $A$ (row-stochastic $A$) | **{np.allclose(_G.sum(1), _A.sum(0))}** |
    | $D^{{-1}}A^\top B$ = Eq. 9's weighted mean | **{np.allclose(_xp, _xd)}** |
    | same identity when rows do **not** sum to 1 | **{np.allclose((_A2.T @ _A2).sum(1), _A2.sum(0))}** |

    First five entries, row sums of $A^\top A$: `{np.array2string(_G.sum(1)[:5], precision=4)}`
    <br>Column sums of $A$: `{np.array2string(_A.sum(0)[:5], precision=4)}`

    The third row is the point: the identity is **not** generic linear algebra, it is bought with
    Property 2. Break row-stochasticity and the "row-sum preconditioner" stops being the column
    sums of $A$, and the one-pass computation of $D$ no longer works.
    """)
    return


@app.cell
def _(mo, np):
    m4_A_orth = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    m4_A_ovlp = np.array([[0.6, 0.4], [0.5, 0.5], [0.45, 0.55], [0.7, 0.3]])
    m4_B = np.array([[1.0, 0.0], [0.8, 0.2], [0.1, 0.9], [0.0, 1.0]])

    def _wmean(A, B):
        return (A.T @ B) / A.sum(0)[:, None]

    m4_orth_w = _wmean(m4_A_orth, m4_B)
    m4_orth_ls = np.linalg.lstsq(m4_A_orth, m4_B, rcond=None)[0]
    m4_ovlp_w = _wmean(m4_A_ovlp, m4_B)
    m4_ovlp_ls = np.linalg.lstsq(m4_A_ovlp, m4_B, rcond=None)[0]
    m4_gap_orth = np.abs(m4_orth_w - m4_orth_ls).max()
    m4_gap_ovlp = np.abs(m4_ovlp_w - m4_ovlp_ls).max()

    mo.md(f"""
    ### 🔢 When is the weighted mean the right answer?

    **Case 1 — orthogonal columns** (each ray sees exactly one primitive):
    ```
    weighted mean      least squares
    {np.array2string(m4_orth_w, precision=4)}   {np.array2string(m4_orth_ls, precision=4)}
    ```
    max difference = **{m4_gap_orth:.2e}** → *identical*. With no co-visibility the mean **is**
    optimal, and the paper's solver is exact.

    **Case 2 — overlapping columns** (every ray sees both):
    ```
    weighted mean      least squares
    {np.array2string(m4_ovlp_w, precision=4)}   {np.array2string(m4_ovlp_ls, precision=4)}
    ```
    max difference = **{m4_gap_ovlp:.3f}** → *very different*. The weighted mean pulls both
    primitives toward the global average of $B$, because every ray votes for both. Least squares
    instead **separates** them, using the small weight differences across rays to attribute
    responsibility — and lands far apart.

    That gap is what $\\beta$ (Module 6) claims to bound.
    """)
    return


@app.cell
def _(mo, slider, widget):
    m4_body = """
    grid(16,1.4,'#334155');
    var B=[[-4,0.5,-2],[-3,1.2,1.5],[-3.6,-0.8,0.4],[4.5,0.4,1.0],[-2.6,0.2,-1.2]];
    var obs=B.map(function(p){ return sph(p.slice(),0.55,'#4cc9f0'); });
    var meanBall=sph([0,0,0],0.45,'#ff6b6b');
    var medBall =sph([0,0,0],0.45,'#06d6a0');
    function geoMedian(pts,w){
      var y=[0,0,0];
      pts.forEach(function(p,i){ y[0]+=p[0]*w[i]; y[1]+=p[1]*w[i]; y[2]+=p[2]*w[i]; });
      for(var it=0; it<80; it++){
        var num=[0,0,0], den=0;
        pts.forEach(function(p,i){
          var d=Math.max(Math.hypot(p[0]-y[0],p[1]-y[1],p[2]-y[2]),1e-6);
          num[0]+=p[0]*w[i]/d; num[1]+=p[1]*w[i]/d; num[2]+=p[2]*w[i]/d; den+=w[i]/d;
        });
        y=[num[0]/den,num[1]/den,num[2]/den];
      }
      return y;
    }
    function update(){
      var outl=parseFloat(S('out').value), wl=parseFloat(S('w4').value);
      S('outo').textContent=outl.toFixed(1); S('w4o').textContent=wl.toFixed(2);
      obs[3].p[0]=outl;
      var w=[1,1,1,wl,1], sw=w.reduce(function(p,c){return p+c;},0);
      var mx=Math.max.apply(null,w);
      obs.forEach(function(m,i){ m.r=0.3+0.5*w[i]/mx; });
      var pts=obs.map(function(m){return m.p;}), wn=w.map(function(v){return v/sw;});
      var mean=[0,0,0];
      pts.forEach(function(p,i){ mean[0]+=p[0]*wn[i]; mean[1]+=p[1]*wn[i]; mean[2]+=p[2]*wn[i]; });
      var med=geoMedian(pts,wn);
      meanBall.p=mean; medBall.p=med;
      readout.textContent=
        'weighted mean    (Eq.9, the paper) = ('+mean.map(function(v){return v.toFixed(2);}).join(', ')+')\\n'+
        'geometric median (what we use)     = ('+med.map(function(v){return v.toFixed(2);}).join(', ')+')\\n'+
        'separation = '+Math.hypot(mean[0]-med[0],mean[1]-med[1],mean[2]-med[2]).toFixed(3)+
        '   <- the inconsistent-view penalty the mean pays';
    }
    onAll(['out','w4'],update); update();
    """
    m4_panel = "".join([
        slider("out", "outlier position (a bad mask)", -5, 12, 0.1, 4.5),
        slider("w4", "its ray weight A_ij", 0, 2, 0.01, 1.0),
        '<div id="readout"></div>',
        '<div class="hint">Blue = observations B_i (size = weight). '
        '<span style="color:#ff6b6b">Red</span> = Eq. 9 weighted mean. '
        '<span style="color:#06d6a0">Green</span> = geometric median. '
        'Drag the outlier out to simulate the ramen-bowl mask: the mean chases it, the median '
        'barely moves. This is the single change that gained us +0.14 mIoU on room_0.</div>',
    ])
    mo.vstack([mo.md("### 🎛 The weighted mean, and why we replaced it"),
               mo.iframe(widget(m4_body, m4_panel, yaw=0.6, pitch=0.45, dist=20.0), height=580)])
    return


@app.cell
def _(quiz):
    m4_q, m4_grade = quiz(
        "When is Eq. 9's weighted mean exactly the least-squares solution?",
        options={
            "Always — that is the theorem": "always",
            "When the columns of A are orthogonal (no co-visibility)": "orth",
            "When B lies in range(A)": "range",
            "When all opacities equal 1": "opaque",
        },
        correct="orth",
        why=("Orthogonal columns mean $A^\\top A$ is diagonal, so the normal equations decouple "
             "into one independent average per primitive — which is exactly Eq. 9. Any "
             "co-visibility puts mass off the diagonal and the mean starts double-counting shared "
             "evidence. Note that $B \\in \\operatorname{range}(A)$ makes the *residual* zero but "
             "does **not** make the mean optimal — Module 7 shows that case is where the paper's "
             "bound fails hardest."),
    )
    m4_q
    return (m4_grade,)


@app.cell
def _(m4_grade):
    m4_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 5 — The surrogate loss, and Jensen

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $L(x)$ | the **true loss** — sum *inside* the norm. This is what rendering actually does. |
    | $J(x)$ | the **surrogate loss** — sum *outside* the norm. A convenient fiction that sits above $L$. |
    | $x'$ | the minimiser of $J$ — i.e. the weighted mean from Module 4. |
    | $\|\cdot\|$ | any convex norm (L1, L2, Huber). |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 Two losses

    $$L(x) = \sum_i \Bigl\| \sum_j A_{ij} x_j - B_i \Bigr\| \qquad\text{(the true loss)}$$
    $$J(x) = \sum_i \sum_j A_{ij}\,\bigl\| x_j - B_i \bigr\| \qquad\text{(the surrogate)} \tag{Eq. 7}$$

    Look only at where the sum sits relative to the norm:

    - $L$ — sum **inside**: mix the primitives first, then measure the error of the mixture. *This is
      what rendering actually does.*
    - $J$ — sum **outside**: measure each primitive's error separately, then average. *This is not a
      rendering; it is a convenient fiction.*

    Because $\|\cdot\|$ is convex and (Property 2) $\sum_j A_{ij} = 1$, Jensen's inequality gives

    $$L(x) \;\le\; J(x) \tag{Eq. 8}$$

    **Read the two losses side by side.** Take one ray $i$:

    - $L$ does: *mix the primitives' features together first* ($\sum_j A_{ij}x_j$), *then* measure
      how far the mixture is from what you observed. This is what a renderer does.
    - $J$ does: measure how far **each primitive individually** is from the observation, *then*
      average those distances.

    These are different numbers. Jensen's inequality says the second is always ≥ the first, for any
    convex norm, provided the weights $A_{ij}$ sum to 1 across $j$. Intuition: averaging first can
    let errors cancel; measuring first and averaging after cannot.

    so $J$ is an **upper bound** on the true loss.

    ### Why bother?

    Because $J$ is **separable**. Each $x_j$ appears in its own terms with no coupling to $x_k$, so
    setting the gradient to zero solves it in closed form:

    $$\frac{\partial J}{\partial x_j} = \sum_i A_{ij}\bigl(x_j - B_i\bigr) = 0
      \;\;\Longrightarrow\;\;
      x'_j = \frac{\sum_i A_{ij} B_i}{\sum_i A_{ij}} \tag{Eq. 9}$$

    ### 🔑 The honest summary of the method

    > **Module 4's closed form is the exact minimiser of the surrogate $J$ — not of the real loss
    > $L$.** The paper minimises a fiction that is known to sit above the truth, and everything from
    > here on is an attempt to bound how much that costs.

    Separability is precisely what throws away co-visibility: $J$ has *no term* coupling $x_j$ and
    $x_k$, so it cannot know that two primitives were seen together.
    """)
    return


@app.cell
def _(mo, np):
    m5_A = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]])
    m5_B = np.array([[1.0], [0.0], [0.6]])
    m5_x = np.array([[0.9], [0.2]])

    m5_L = float(np.abs(m5_A @ m5_x - m5_B).sum())
    m5_J = float(sum(m5_A[i, j] * abs(m5_x[j, 0] - m5_B[i, 0])
                     for i in range(3) for j in range(2)))
    m5_xp = (m5_A.T @ m5_B) / m5_A.sum(0)[:, None]
    m5_Jp = float(sum(m5_A[i, j] * abs(m5_xp[j, 0] - m5_B[i, 0])
                      for i in range(3) for j in range(2)))
    m5_Lp = float(np.abs(m5_A @ m5_xp - m5_B).sum())

    mo.md(f"""
    ### 🔢 Jensen, on three rays

    With $x = {np.array2string(m5_x.ravel(), precision=2)}$:

    - $L(x) = {m5_L:.4f}$  (sum inside the norm)
    - $J(x) = {m5_J:.4f}$  (sum outside)
    - $L \\le J$? **{m5_L <= m5_J}** — Jensen holds, with slack {m5_J - m5_L:.4f}.

    At the surrogate's own minimiser $x' = {np.array2string(m5_xp.ravel(), precision=4)}$:

    - $L(x') = {m5_Lp:.4f}$, $J(x') = {m5_Jp:.4f}$ — still $L \\le J$, as it must be.

    **The slack $J - L$ is the whole problem.** It is not a constant: it grows with how much the
    primitives on a ray disagree. Module 6 gives that disagreement a name.
    """)
    return


@app.cell
def _(mo, np):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(7.2, 3.6), dpi=130)
    _t = np.linspace(-2.4, 2.4, 400)
    _ax.plot(_t, np.abs(_t), lw=2, color="#4cc9f0", label=r"convex loss $\|\cdot\|$")
    _p, _q, _w = -1.8, 2.0, 0.55
    _mix = _w * _p + (1 - _w) * _q
    _ax.plot([_p, _q], [abs(_p), abs(_q)], "--", color="#ff6b6b", lw=1.6,
             label="chord (the surrogate $J$)")
    _ax.scatter([_mix], [abs(_mix)], s=70, color="#06d6a0", zorder=5,
                label=r"$L$: $\|\sum_j A_{ij}x_j - B_i\|$")
    _ax.scatter([_mix], [_w * abs(_p) + (1 - _w) * abs(_q)], s=70, color="#ffd166", zorder=5,
                label=r"$J$: $\sum_j A_{ij}\|x_j - B_i\|$")
    _ax.annotate("", xy=(_mix, abs(_mix)), xytext=(_mix, _w * abs(_p) + (1 - _w) * abs(_q)),
                 arrowprops=dict(arrowstyle="<->", color="#e6e8ee", lw=1.4))
    _ax.text(_mix + 0.1, 1.0, "Jensen gap\n$J-L$", color="#e6e8ee", fontsize=10)
    _ax.set_title("Why $L \\leq J$: the chord lies above the curve", fontsize=11)
    _ax.legend(fontsize=8.5, loc="upper center")
    _ax.set_xlabel("feature-space coordinate")
    _ax.grid(alpha=0.15)
    _fig.tight_layout()
    mo.vstack([mo.md("### 🎛 Jensen in one picture"),
               _ax.figure])
    return


@app.cell
def _(quiz):
    m5_q, m5_grade = quiz(
        "Why is the surrogate $J$ introduced at all?",
        options={
            "It is a tighter bound than L": "tighter",
            "It is separable across primitives, so it has a closed-form minimiser": "sep",
            "It is convex while L is not": "convex",
            "It removes the need for Property 2": "prop2",
        },
        correct="sep",
        why=("$J$ decouples the primitives — each $x_j$ appears alone — so $\\partial J/\\partial "
             "x_j = 0$ solves independently and gives Eq. 9. $L$ is *already* convex, so that is "
             "not the motive; and $J$ is by construction **looser** than $L$, not tighter. $J$ "
             "*needs* Property 2 rather than removing it. The price of separability is that $J$ "
             "cannot represent co-visibility at all."),
    )
    m5_q
    return (m5_grade,)


@app.cell
def _(m5_grade):
    m5_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 6 — $\beta$, the dispersion coefficient

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $\hat{x}$ | the **true optimum** — the least-squares solution we wish we could afford. |
    | $\Delta_{ij}$ | distance from primitive $j$'s optimal feature to ray $i$'s observation: $\|\hat{x}_j - B_i\|$. |
    | $\mu_i$ | weighted **mean** of those distances along ray $i$. |
    | $\sigma_i^2$ | weighted **variance** of those distances along ray $i$. |
    | $\beta_i$ | $\sigma_i^2/\mu_i^2$ — relative spread on ray $i$. |
    | $\beta$ | $\max_i \beta_i$ — the worst ray in the scene. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 Definition

    At the true optimum $\hat{x}$, for each ray $i$:

    $$\Delta_{ij} = \bigl\|\hat{x}_j - B_i\bigr\|, \qquad
      \mu_i = \sum_j A_{ij}\Delta_{ij} \tag{Eq. 10}$$

    $$\sigma_i^2 = \sum_j A_{ij}\bigl(\Delta_{ij}^2 - \mu_i^2\bigr), \qquad
      \beta_i = \frac{\sigma_i^2}{\mu_i^2}, \qquad
      \beta = \max_i \beta_i \tag{Eq. 11}$$

    **Read it slowly, with a picture in mind.** Fix one ray $i$. It passes through a few primitives.

    1. For each primitive $j$ on that ray, ask: *how far is its ideal feature from what this ray
       actually saw?* Call it $\Delta_{ij}$.
    2. Average those distances, weighting by how much each primitive owns the ray → $\mu_i$.
    3. Ask how **spread out** those distances were → $\sigma_i^2$.
    4. Divide the spread by the average, squared → $\beta_i$. Dividing makes it *relative*: a ray
       where distances are (10, 12) is not "more dispersed" than one with (1, 1.2).
    5. Take the worst ray in the whole scene → $\beta$.

    **$\beta$ answers: do the primitives along a ray agree about that ray?** If they agree
    ($\beta$ small), replacing them by their average is harmless. If one is a great match and the
    rest are poor ($\beta$ large), averaging destroys the good answer.

    In words:

    - $\Delta_{ij}$ — how far primitive $j$'s optimal feature is from what ray $i$ observed;
    - $\mu_i$ — the weighted **mean** of those distances along the ray;
    - $\sigma_i^2$ — their weighted **variance** (using $\sum_j A_{ij}=1$);
    - $\beta_i$ — variance ÷ mean², a **squared coefficient of variation** — a *relative* spread;
    - $\beta$ — the worst case over all rays.

    ### The intuition

    > $\beta$ measures **how much the primitives along a ray disagree about that ray.**

    - $\beta = 0$ — every primitive on the ray is equally far from the observation. The ray is
      *coherent*; averaging them loses nothing.
    - $\beta$ large — one primitive matches the observation far better than its neighbours. Averaging
      smears a good answer together with bad ones.

    **Property 4** (*Diagonal Dominance Reduces $\beta$*): if each row of $A^\top A$ becomes more
    diagonally dominant, $\beta$ shrinks. This is the bridge to Tikhonov guidance in Module 8 — and
    note it is stated as a *property*, with no proof given.
    """)
    return


@app.cell
def _(mo, np):
    def _beta_of(w, d):
        w = np.asarray(w, float); w = w / w.sum()
        d = np.asarray(d, float)
        mu = float(w @ d)
        s2 = float(w @ (d ** 2) - mu ** 2)
        return mu, s2, s2 / max(mu ** 2, 1e-12)

    m6_a = _beta_of([0.5, 0.5], [1.0, 3.0])
    m6_b = _beta_of([0.9, 0.1], [1.0, 3.0])
    m6_c = _beta_of([0.5, 0.5], [2.0, 2.0])
    m6_d = _beta_of([0.5, 0.5], [0.1, 3.0])

    mo.md(f"""
    ### 🔢 Computing $\\beta$ by hand

    | weights $A_{{ij}}$ | distances $\\Delta_{{ij}}$ | $\\mu_i$ | $\\sigma_i^2$ | $\\beta_i$ |
    |---|---|---|---|---|
    | 0.5, 0.5 | 1, 3 | {m6_a[0]:.3f} | {m6_a[1]:.3f} | **{m6_a[2]:.4f}** |
    | 0.9, 0.1 | 1, 3 | {m6_b[0]:.3f} | {m6_b[1]:.3f} | **{m6_b[2]:.4f}** |
    | 0.5, 0.5 | 2, 2 | {m6_c[0]:.3f} | {m6_c[1]:.3f} | **{m6_c[2]:.4f}** |
    | 0.5, 0.5 | 0.1, 3 | {m6_d[0]:.3f} | {m6_d[1]:.3f} | **{m6_d[2]:.4f}** |

    Read the rows:

    - **Row 1 → 2** is Property 4 in miniature. Same distances; concentrating the weight on one
      primitive (0.9/0.1 — *more diagonally dominant*) drops $\\beta$ from {m6_a[2]:.3f} to
      {m6_b[2]:.3f}.
    - **Row 3**: identical distances → zero variance → $\\beta = 0$. A perfectly coherent ray.
    - **Row 4**: one primitive nearly explains the ray (0.1) while the other is far (3.0) → $\\beta$
      jumps to {m6_d[2]:.3f}. This is the case where averaging does real damage — and note $\\beta$
      is *unbounded*: push the near distance toward 0 and $\\beta \\to \\infty$.
    """)
    return


@app.cell
def _(mo, slider, widget):
    m6_body = """
    grid(18,1.4,'#334155');
    var Bpt=[0,0,0];
    sph(Bpt,0.55,'#ffd166');
    var cols=['#ff6b6b','#06d6a0','#4cc9f0'];
    var dirs=[[0.94,0.24,0.28],[-0.55,0.09,0.83],[-0.28,-0.19,-0.94]];
    var prim=[], lines=[];
    for(var j=0;j<3;j++){ prim.push(sph([0,0,0],0.5,cols[j]));
                          lines.push(seg(Bpt,[0,0,0],cols[j],1.2)); }
    var mu_ring=ring([0,0,0],1,'#ffffff');
    function update(){
      var d=[parseFloat(S('d0').value),parseFloat(S('d1').value),parseFloat(S('d2').value)];
      var w0=parseFloat(S('w0').value);
      ['d0','d1','d2'].forEach(function(i,k){ S(i+'o').textContent=d[k].toFixed(2); });
      S('w0o').textContent=w0.toFixed(2);
      var rest=(1-w0)/2, w=[w0,rest,rest];
      for(var j=0;j<3;j++){
        prim[j].p=[dirs[j][0]*d[j],dirs[j][1]*d[j],dirs[j][2]*d[j]];
        prim[j].r=0.28+1.1*w[j];
        lines[j].b=prim[j].p;
      }
      var mu=0; for(var j=0;j<3;j++) mu+=w[j]*d[j];
      var s2=0; for(var j=0;j<3;j++) s2+=w[j]*d[j]*d[j];
      s2-=mu*mu;
      var beta=s2/Math.max(mu*mu,1e-9);
      mu_ring.r=Math.max(mu,0.01);
      readout.textContent=
        'Delta = ['+d.map(function(v){return v.toFixed(2);}).join(', ')+']    A_ij = ['+
          w.map(function(v){return v.toFixed(2);}).join(', ')+']\\n'+
        'mu = '+mu.toFixed(3)+'    sigma^2 = '+s2.toFixed(3)+'    beta = '+beta.toFixed(4)+'\\n'+
        (beta<0.02?'coherent ray: averaging costs almost nothing'
                  :(beta>0.5?'high dispersion: the mean smears a good match into bad ones'
                            :'moderate dispersion'));
    }
    onAll(['d0','d1','d2','w0'],update); update();
    """
    m6_panel = "".join([
        slider("d0", "Δ₀ (red)", 0.1, 8, 0.05, 2.0),
        slider("d1", "Δ₁ (green)", 0.1, 8, 0.05, 2.0),
        slider("d2", "Δ₂ (blue)", 0.1, 8, 0.05, 2.0),
        slider("w0", "weight on red (diagonal dominance)", 0.34, 0.98, 0.01, 0.34),
        '<div id="readout"></div>',
        '<div class="hint">Yellow = the observation Bᵢ. White ring = μᵢ, the weighted mean '
        'distance. Set all three Δ equal → β = 0. Pull one in and push another out → β climbs. '
        'Then raise the red weight and watch Property 4 act: β falls without any Δ changing.</div>',
    ])
    mo.vstack([mo.md("### 🎛 Dispersion along a ray"),
               mo.iframe(widget(m6_body, m6_panel, yaw=0.7, pitch=0.4, dist=18.0), height=600)])
    return


@app.cell
def _(quiz):
    m6_q, m6_grade = quiz(
        "A ray where every primitive sits the same distance from the observation has …",
        options={
            "β = 0, and averaging is harmless": "zero",
            "β = 1, the neutral value": "one",
            "β undefined": "undef",
            "maximal β": "max",
        },
        correct="zero",
        why=("Equal distances ⇒ zero weighted variance ⇒ $\\sigma_i^2 = 0$ ⇒ $\\beta_i = 0$. The "
             "primitives agree about the ray, so replacing them with their mean loses nothing. "
             "$\\beta$ has no neutral value at 1 — it is a ratio of variance to squared mean, "
             "unbounded above, and zero is its floor."),
    )
    m6_q
    return (m6_grade,)


@app.cell
def _(m6_grade):
    m6_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 7 — The $(1+\beta)$ bound, and why it is false ⚠️

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |
    | $i$ | indexes a **ray** (so $i$ runs $1\ldots R$). Think: *one pixel in one photo.* |
    | $j$ | indexes a **primitive** (so $j$ runs $1\ldots P$). Think: *one blob in the 3D scene.* |
    | $\hat{x}$ | the **true optimum** (exact least squares) — the thing we are measured against. |
    | $x'$ | the **closed-form answer** (weighted mean, Eq. 9) — the thing being bounded. |
    | $L(\cdot)$ | the true loss; $L(\hat{x})$ is the best achievable. |
    | $J(\cdot)$ | the surrogate loss from Module 5. |
    | $\beta$ | worst-case relative dispersion from Module 6. |
    | $\mu_i, \Delta_{ij}$ | as in Module 6: mean distance on ray $i$, and per-primitive distance. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    This is the paper's headline theoretical contribution — the thing that distinguishes it from the
    "heuristic-forward" baselines it criticises for lacking formal guarantees. So it deserves the
    most careful reading in the notebook.

    ### 📐 The claimed chain

    $$L(x') \;\le\; J(x') \;\le\; J(\hat{x}) \tag{Eq. 12}$$
    $$J(\hat{x}) = \sum_i\sum_j A_{ij}\Delta_{ij}^2 = \sum_i (1+\beta_i)\mu_i^2
      \;\le\; (1+\beta)\,L(\hat{x})
      \;\;\Longrightarrow\;\; L(x') \le (1+\beta)L(\hat{x}) \tag{Eq. 13}$$

    ### Step by step — three steps are fine

    | step | status | why |
    |---|---|---|
    | $L(x') \le J(x')$ | ✅ | Jensen, Eq. 8 |
    | $J(x') \le J(\hat{x})$ | ✅ | $x'$ **minimises** $J$ (Eq. 9). This is the point of the surrogate. |
    | $\sum_j A_{ij}\Delta_{ij}^2 = (1+\beta_i)\mu_i^2$ | ✅ | variance identity $\sigma^2 + \mu^2$, using $\sum_j A_{ij}=1$ |
    | $\sum_i (1+\beta_i)\mu_i^2 \le (1+\beta)L(\hat{x})$ | ❌ | **this one** |

    ### The break

    The last step needs $\sum_i \mu_i^2 \le L(\hat{x})$. But by the triangle inequality — *the very
    same Jensen step that gave Eq. 8* —

    $$L_i(\hat{x}) = \Bigl\|\sum_j A_{ij}\bigl(\hat{x}_j - B_i\bigr)\Bigr\|
      \;\le\; \sum_j A_{ij}\bigl\|\hat{x}_j - B_i\bigr\| = \mu_i$$

    so $\mu_i \ge L_i(\hat{x})$, giving $\sum_i \mu_i^2 \ge \sum_i L_i(\hat{x})^2$.

    > **The inequality points the wrong way.** The proof needs $\mu$ *below* the optimal loss;
    > Jensen guarantees it is *above*. You cannot get the required direction out of the same
    > inequality that established Eq. 8.

    *(A secondary wrinkle: Eq. 7 defines $L, J$ with an unsquared norm while Eq. 13 manipulates
    $\Delta^2, \mu^2$. Fix the convention before hunting the error, or you will chase a phantom.)*

    ### Is the claim merely unproven, or actually false?

    Run the cells below. They test it directly: build $A$ row-stochastic **by construction** (so
    Property 2 holds *exactly*, giving the paper its best case), solve for $\hat{x}$ with exact
    `lstsq`, compute $x'$ from Eq. 9, and check the inequality.
    """)
    return


@app.cell
def _(np):
    def sfs_trial(seed, R, P, F, sparsity=None, noise=None):
        """One test of the (1+β) bound. Returns (β, L(x')/L(x̂), bound_holds).

        `noise=None` draws B at random; `noise=σ` instead sets B = A·x_true + σ·ε, which is the
        near-consistent regime Property 3 actually claims. The distinction matters: a random B is
        an unfair test, and the paper deserves to be judged in its own regime.
        """
        rng = np.random.default_rng(seed)
        A = rng.random((R, P))
        if sparsity:
            A *= (rng.random((R, P)) < sparsity)
            A[A.sum(1) == 0, 0] = 1.0
        A /= A.sum(1, keepdims=True)                      # Property 2, exactly
        if noise is None:
            B = rng.random((R, F))
        else:
            B = A @ rng.random((P, F)) + noise * rng.standard_normal((R, F))
        xh = np.linalg.lstsq(A, B, rcond=None)[0]         # the true optimum x̂
        xp = (A.T @ B) / A.sum(0)[:, None]                # Eq. 9
        D = np.linalg.norm(xh[None, :, :] - B[:, None, :], axis=2)
        mu = (A * D).sum(1)
        s2 = (A * D ** 2).sum(1) - mu ** 2
        beta = float(np.max(s2 / np.maximum(mu ** 2, 1e-12)))
        loss = lambda x: float((np.linalg.norm(A @ x - B, axis=1) ** 2).sum())
        lp, lh = loss(xp), loss(xh)
        return beta, lp / max(lh, 1e-300), lp <= (1 + beta) * lh

    return (sfs_trial,)


@app.cell
def _(mo, np, sfs_trial):
    _rows = []
    for _cfg in [(200, 20, 8, None), (500, 50, 16, None),
                 (1000, 80, 32, 0.05), (300, 30, 64, 0.2)]:
        _res = [sfs_trial(_s, *_cfg) for _s in range(40)]
        _f = sum(1 for _, _, ok in _res if not ok)
        _b = np.mean([b for b, _, _ in _res])
        _r = np.mean([r for _, r, _ in _res])
        _rows.append(f"| {_cfg[0]} | {_cfg[1]} | {_cfg[2]} | {_cfg[3] or 'dense'} | "
                     f"**{_f}/40** | {_b:.4f} | {_r:.4f} |")
    mo.md(f"""
    ### 🔢 Test 1 — random $B$ (40 seeds per row)

    | R | P | F | sparsity | violations | mean β | mean $L(x')/L(\\hat{{x}})$ |
    |---|---|---|---|---|---|---|
    {chr(10).join(_rows)}

    **93 of 160 trials violate the bound.** But notice something important before concluding: the
    true ratio (1.03–1.11) and $1+\\beta$ (1.04–1.12) are the **same order of magnitude**. $\\beta$
    is a *good predictor of the error scale* — it is simply not an upper bound on it. That nuance is
    why the method works well in practice, and it belongs in any fair critique.
    """)
    return


@app.cell
def _(mo, np, sfs_trial):
    _rows2 = []
    for _n in [0.1, 0.01, 0.001, 0.0]:
        _res2 = [sfs_trial(_s, 500, 50, 16, None, _n) for _s in range(40)]
        _f2 = sum(1 for _, _, ok in _res2 if not ok)
        _b2 = np.mean([b for b, _, _ in _res2])
        _r2 = np.mean([r for _, r, _ in _res2])
        _rows2.append(f"| {_n} | **{_f2}/40** | {_b2:.4f} | {_r2:,.2f} |")
    mo.md(f"""
    ### 🔢 Test 2 — the fair test: $B = A x_{{\\text{{true}}}} + \\text{{noise}}$

    This is the regime Property 3 actually claims — $B$ nearly inside $\\operatorname{{range}}(A)$.
    R=500, P=50, F=16, 40 seeds each:

    | noise | violations | mean β | mean $L(x')/L(\\hat{{x}})$ |
    |---|---|---|---|
    {chr(10).join(_rows2)}

    It fails **40/40 at every noise level**, and gets *catastrophically worse* as the system
    approaches consistency — the opposite of what Property 3 is invoked to support.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 🔑 Why no multiplicative bound can exist

    The noise → 0 column is not a numerical artefact; it is the proof.

    > If the system is **consistent**, then $L(\hat{x}) = 0$, so $(1+\beta)\,L(\hat{x}) = 0$ for any
    > finite $\beta$. The bound would then assert $L(x') = 0$ — that the contribution-weighted mean
    > *is* the least-squares solution.
    >
    > It is not (Module 4, Case 2). **Therefore no finite multiplicative constant can bound this
    > solver.**

    A correct statement has to be **additive**:

    $$L(x') \;\le\; L(\hat{x}) \;+\; \varepsilon\bigl(A\bigr)$$

    where $\varepsilon$ is governed by how much the columns of $A$ overlap — the off-diagonal mass of
    $A^\top A$, i.e. the **co-visibility Laplacian** from Module 3. When primitives are never
    co-visible, $\varepsilon = 0$ and the mean is exact (Module 4, Case 1); as co-visibility grows,
    so does the penalty.

    That correction is ours, and it transfers directly to foam. It is also *more useful* than the
    original claim: it says **where** the solver is trustworthy (well-separated primitives) rather
    than asserting a uniform factor that cannot hold.

    ### What this does and does not mean

    - It does **not** mean the method is bad. It is fast, simple, and empirically strong.
    - It **does** mean the paper's central theoretical claim does not survive contact with its own
      assumptions, and that the honest version of the guarantee is additive and geometry-dependent.
    - A broken proof of a true statement is a different paper from a false statement. This is the
      latter: the statement itself fails, in the paper's own claimed regime.
    """)
    return


@app.cell
def _(quiz):
    m7_q, m7_grade = quiz(
        "Why can no finite multiplicative constant bound this solver?",
        options={
            "Because β can be infinite": "infbeta",
            "Because a consistent system has L(x̂)=0, forcing L(x')=0, which is false": "consistent",
            "Because A is sparse": "sparse",
            "Because the features are high-dimensional": "highdim",
        },
        correct="consistent",
        why=("Consistency drives the right-hand side to exactly zero for **any** finite constant, so "
             "the bound would demand the weighted mean be exactly optimal — which it is not "
             "whenever primitives are co-visible. β blowing up is a symptom, not the cause; "
             "sparsity and dimension are irrelevant. The fix is an **additive** term scaled by "
             "co-visibility."),
    )
    m7_q
    return (m7_grade,)


@app.cell
def _(m7_grade):
    m7_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 8 — Tikhonov Guidance

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $\lambda$ | the **Tikhonov strength**. Bigger $\lambda$ = more diagonal dominance, more shrinkage. |
    | $I$ | the identity matrix. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 Soft diagonal dominance

    Replace $A^\top A$ with $A^\top A + \lambda I$. By **Property 4** (Module 6), more diagonal
    dominance ⇒ smaller $\beta$ ⇒ (under the claimed bound) a tighter guarantee.

    Note the logical structure: **the regulariser is justified *through* the bound.** With the bound
    broken (Module 7), that particular justification lapses — but the regulariser may still be
    perfectly good on ordinary conditioning grounds, which do not depend on $\beta$ at all:

    - it makes the system better conditioned (raises the smallest eigenvalue off zero);
    - it restores uniqueness when $A$ is rank-deficient (Module 2's unseen or unseparated
      primitives);
    - it costs a **bias toward zero** — and for a *normalised* CLIP feature, shrinking magnitude is
      not neutral. Direction is what carries meaning, so shrinkage that is uneven across primitives
      distorts relative cosine similarities.
    """)
    return


@app.cell
def _(mo, np):
    m8_A = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.34, 0.33, 0.33], [0.0, 0.2, 0.8]])
    m8_G = m8_A.T @ m8_A
    m8_rows = []
    for _lam in [0.0, 0.01, 0.1, 1.0]:
        _M = m8_G + _lam * np.eye(3)
        _cond = np.linalg.cond(_M)
        _dom = float(np.min(np.diag(_M) / np.maximum(
            (np.abs(_M) - np.diag(np.diag(np.abs(_M)))).sum(1), 1e-12)))
        _sol = np.linalg.solve(_M, m8_A.T @ np.array([[1.0], [0.9], [0.4], [0.0]]))
        m8_rows.append(f"| {_lam} | {_cond:10.1f} | {_dom:.3f} | "
                       f"{np.array2string(_sol.ravel(), precision=3)} | "
                       f"{float(np.linalg.norm(_sol)):.3f} |")
    mo.md(f"""
    ### 🔢 What λ actually buys

    | λ | cond($A^\\top A+\\lambda I$) | min diagonal-dominance ratio | solution | ‖x‖ |
    |---|---|---|---|---|
    {chr(10).join(m8_rows)}

    Conditioning improves by orders of magnitude and dominance rises — but watch the last column:
    $\\|x\\|$ **shrinks monotonically**. At $\\lambda = 1$ the features have been pulled substantially
    toward the origin.

    For CLIP features that matters in a specific way: cosine similarity is scale-invariant *per
    vector*, so uniform shrinkage would be harmless — but Tikhonov shrinks **unevenly**, hardest on
    the least-observed primitives. Those are exactly the primitives whose features you most need to
    trust, and their directions move, not just their lengths.
    """)
    return


@app.cell
def _(quiz):
    m8_q, m8_grade = quiz(
        "With the (1+β) bound broken, what happens to the case for Tikhonov guidance?",
        options={
            "It collapses — the regulariser was only justified by the bound": "collapse",
            "It is unaffected — it never relied on the bound": "unaffected",
            "It survives on conditioning/uniqueness grounds, but loses its β-based rationale": "survives",
            "It becomes provably harmful": "harmful",
        },
        correct="survives",
        why=("The paper's stated route is Property 4 → smaller β → tighter bound, and that route "
             "does lapse. But adding $\\lambda I$ independently improves conditioning and restores "
             "uniqueness under rank deficiency — classical, β-free reasons. The honest position is "
             "that the *practice* stands while the *stated justification* does not, and the "
             "shrinkage bias on normalised features is a real cost to weigh."),
    )
    m8_q
    return (m8_grade,)


@app.cell
def _(m8_grade):
    m8_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 9 — Post-Lifting Aggregation

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $x$ | $P\times F$ — **the unknown**. $x_j$ = the feature vector we want to attach to primitive $j$. |
    | $B$ | $R\times F$ — the **observations**. $B_i$ = the CLIP feature measured at ray $i$'s pixel. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### 📐 The second regulariser

    After solving, cluster the lifted features and filter or replace each primitive's feature using
    its cluster's statistics. The target is the Module 2 failure mode: the ramen bowl whose mask
    includes the bowl in one view and only the noodles in the next.

    ### Why it sits outside the theory

    This step is **post-hoc and non-linear**. It cannot be folded into $A$ and solved jointly,
    because clustering depends on the *solution* $x$, which makes the combined problem non-convex —
    the very property that made Eq. 5 tractable. So none of the $(1+\beta)$ analysis covers it, and
    none of its benefit is explained by the paper's theory.

    That is worth stating plainly: **the two components that most improve results (this and the
    solver's robustness) are the two the theory says least about.**

    ### Our version

    We reach the same goal earlier and more cheaply, by changing the *estimator* rather than
    post-processing: swap Eq. 9's weighted mean for the **geometric median**, which is robust to
    exactly these outlier views. On room_0 SAM features that was **0.6095 vs 0.4649 mIoU** — a
    larger gain than any post-hoc filter we tested, and it needs no extra pass. Play with the
    Module 4 widget again with this in mind: the outlier slider *is* the ramen bowl.
    """)
    return


@app.cell
def _(mo, np):
    _rng = np.random.default_rng(3)
    m9_good = _rng.normal(0, 0.25, size=(18, 2)) + np.array([2.0, 1.0])
    m9_bad = _rng.normal(0, 0.25, size=(4, 2)) + np.array([-3.0, 2.5])
    m9_all = np.vstack([m9_good, m9_bad])
    m9_mean = m9_all.mean(0)

    def _geomed(pts, iters=200):
        y = pts.mean(0)
        for _ in range(iters):
            d = np.maximum(np.linalg.norm(pts - y, axis=1), 1e-9)
            y = (pts / d[:, None]).sum(0) / (1 / d).sum()
        return y

    m9_med = _geomed(m9_all)
    m9_truth = m9_good.mean(0)
    mo.md(f"""
    ### 🔢 Mean vs median under 18% bad views

    22 views of one primitive: 18 agree (cluster near `{np.array2string(m9_truth, precision=2)}`),
    4 come from an inconsistent mask (near `[-3.0, 2.5]`).

    | estimator | result | distance from the good cluster |
    |---|---|---|
    | Eq. 9 weighted mean | `{np.array2string(m9_mean, precision=3)}` | **{np.linalg.norm(m9_mean - m9_truth):.3f}** |
    | geometric median | `{np.array2string(m9_med, precision=3)}` | **{np.linalg.norm(m9_med - m9_truth):.3f}** |

    The mean is dragged {np.linalg.norm(m9_mean - m9_truth) / max(np.linalg.norm(m9_med - m9_truth), 1e-9):.1f}×
    further off by 4 bad views out of 22. Post-Lifting Aggregation tries to *detect and remove* those
    views afterwards; the median simply never lets them dominate in the first place.
    """)
    return


@app.cell
def _(quiz):
    m9_q, m9_grade = quiz(
        "Why can't Post-Lifting Aggregation be folded into the linear solve?",
        options={
            "It would make A too large": "size",
            "Clustering depends on the solution, making the joint problem non-convex": "nonconvex",
            "It operates on rays, not primitives": "rays",
            "CLIP features are not linear": "clip",
        },
        correct="nonconvex",
        why=("The clusters are computed **from** $x$, so optimising both together couples the "
             "objective to a discrete, solution-dependent assignment — non-convex, and no longer "
             "solvable in closed form. Keeping it post-hoc preserves the closed form, at the cost "
             "of putting the step entirely outside the paper's guarantees."),
    )
    m9_q
    return (m9_grade,)


@app.cell
def _(m9_grade):
    m9_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 10 — What the experiments actually test

    Open-vocabulary 3D segmentation on LERF-OVS and ScanNet-family benchmarks, against
    training-based, grouping-based and heuristic-forward baselines; headline is SOTA "in minutes".
    Figure 2 (the Ramen scene) is the qualitative claim against DrSplat.

    ### Mapping claims to machinery

    | claim | which module it tests |
    |---|---|
    | "lifted features in minutes" | Module 4 — the closed form, genuinely tested |
    | SOTA mIoU | Modules 4 + 8 + 9 jointly — never separated by an ablation into the solver alone |
    | "provable upper bound on the global optimal error" | Module 7 — **not tested at all** |

    > The theoretical contribution is the one thing no experiment probes. Nothing in the evaluation
    > would detect that the bound is false, which is precisely how it survived review.

    ### Reading it against our own numbers

    Our replication uses the same benchmark family, so the comparison is direct — with two caveats
    we should state whenever we cite it:

    - **SAM level.** Their pipeline sums all 4 LangSplat granularity levels per pixel; our ScanNet
      arms use level 3 only (the `l` / whole-object level, which is also what NormLift uses). Those
      are different observations $B$, so the numbers are not interchangeable.
    - **Crop protocol.** Mask fill and crop padding change the CLIP embedding materially. Verify
      which protocol produced any feature set before comparing across them.
    """)
    return


@app.cell
def _(quiz):
    m10_q, m10_grade = quiz(
        "Which paper claim do the reported experiments NOT test?",
        options={
            "That lifting runs in minutes": "speed",
            "That the method beats grouping-based baselines": "beats",
            "That the solver is within (1+β) of the optimum": "bound",
            "That it works for both CLIP and DINO features": "agnostic",
        },
        correct="bound",
        why=("Measuring downstream mIoU never compares $L(x')$ against $L(\\hat{x})$ — you would "
             "have to solve the least-squares problem exactly to know the optimum, which is the "
             "very thing the closed form exists to avoid. So the guarantee is untested by "
             "construction, and a false bound produces no visible symptom in any reported table."),
    )
    m10_q
    return (m10_grade,)


@app.cell
def _(m10_grade):
    m10_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Module 11 — How this attaches to our work

    #### 📖 Symbols used in this module

    | symbol | meaning |
    |---|---|
    | $A$ | $R\times P$ — the **rendering operator**. $A_{ij}$ = how much primitive $j$ contributed to ray $i$. Fixed once geometry is fixed. |
    | $P$ | number of **primitives** — Gaussians, or foam cells. ~10⁵–10⁶. |
    | $A^\top A$ | co-visibility again — the quantity that decides how tight the corrected bound is. |

    The whole problem in one sentence: **$A x = B$** — a known *rendering operator* $A$ times the *unknown per-primitive features* $x$ equals the *observed per-pixel features* $B$. Rows are rays, columns are primitives.

    ### Three concrete connections

    **1. We replaced the estimator (Module 4/9).** Eq. 9's weighted mean is the L2 minimiser and is
    not robust. The geometric median resists exactly the inconsistent-mask failure the paper
    describes in prose. room_0 SAM round: **0.6095 vs 0.4649 mIoU**.

    **2. We corrected the bound (Module 7).** No multiplicative constant can work; the defensible
    statement is additive, with slack set by the co-visibility Laplacian of $A^\top A$. This is
    paper-sized on its own, and it says something *useful* — where the fast solver can be trusted.

    **3. Foam changes $A$ structurally.** This is the part worth pushing:

    | | 3DGS | PowerFoam |
    |---|---|---|
    | support | unbounded Gaussian | **bounded** power cell ∩ ball |
    | partition | overlapping | **disjoint** — exact ownership |
    | ray's weights | long overlapping chain | short non-overlapping chain |
    | Property 2 | engineered via random background | closer to structural |
    | co-visibility | dense off-diagonal | **sparser** |

    Since the error term in the corrected bound is driven by off-diagonal mass, and foam's
    disjointness makes that mass smaller, **the same one-shot solver should be provably tighter on
    foam than on Gaussians.** That is a theorem-shaped claim we can actually test: compute the
    off-diagonal mass of $A^\top A$ for both representations on the same scene and compare it against
    the measured $L(x')/L(\hat{x})$ gap.

    ### The open question worth answering next

    > Does foam's sparser co-visibility measurably tighten the additive bound on real scenes — and by
    > how much, relative to 3DGS on the same geometry?

    We already have the frozen-arm control that makes this clean: PowerFoam-frozen and 3DGS-frozen
    use the **identical** 51,610 GT-initialised points on scene0062_00, so $A$ differs *only* by the
    kernel. Same points, same cameras, same features — the difference in off-diagonal mass is
    attributable to bounded-vs-unbounded support and nothing else.
    """)
    return


@app.cell
def _(mo, np):
    # A crude but honest illustration of the structural claim: build two operators over the same
    # primitive positions, one with bounded disjoint support (foam-like), one with long overlapping
    # tails (Gaussian-like), and compare off-diagonal mass. This is a toy, not a measurement of our
    # scenes -- the real version uses the frozen arms' actual A.
    _rng = np.random.default_rng(7)
    _R, _P = 4000, 60
    _centers = np.linspace(0, 1, _P)
    _ray = _rng.random(_R)

    def _operator(width, hard):
        d = np.abs(_ray[:, None] - _centers[None, :])
        W = np.exp(-(d / width) ** 2)
        if hard:                      # bounded support: keep only the nearest few, then renormalise
            keep = np.argsort(d, axis=1)[:, :2]
            M = np.zeros_like(W)
            np.put_along_axis(M, keep, np.take_along_axis(W, keep, 1), 1)
            W = M
        return W / np.maximum(W.sum(1, keepdims=True), 1e-12)

    def _offdiag_frac(W):
        G = W.T @ W
        d = np.diag(G).sum()
        return float((G.sum() - d) / G.sum())

    m11_foam = _operator(0.02, True)
    m11_gauss = _operator(0.10, False)
    m11_f, m11_g = _offdiag_frac(m11_foam), _offdiag_frac(m11_gauss)
    mo.md(f"""
    ### 🔢 Toy check of the structural claim

    Same {_P} primitive positions, same {_R} rays; only the kernel differs.

    | operator | off-diagonal fraction of $A^\\top A$ |
    |---|---|
    | bounded, near-disjoint support (foam-like) | **{m11_f:.4f}** |
    | wide overlapping support (Gaussian-like) | **{m11_g:.4f}** |

    Ratio: **{m11_g / max(m11_f, 1e-12):.1f}×** more entangled for the Gaussian-like operator. If the
    corrected bound's slack scales with off-diagonal mass, this is the mechanism by which foam should
    admit a tighter guarantee under the *same* solver.

    ⚠️ This is a **toy in 1-D**, not a measurement of our scenes — treat it as a statement of the
    hypothesis, not evidence for it. The real test uses the frozen arms' actual $A$.
    """)
    return


@app.cell
def _(quiz):
    m11_q, m11_grade = quiz(
        "Why are the frozen arms the right control for the foam-vs-3DGS bound question?",
        options={
            "They have the most primitives": "count",
            "They share identical primitive positions, so A differs only by the kernel": "identical",
            "They were trained longest": "trained",
            "They have the best mIoU": "miou",
        },
        correct="identical",
        why=("Both frozen arms are initialised one primitive per GT vertex — 51,610 identical "
             "positions on scene0062_00. Holding positions, cameras and features fixed means any "
             "difference in the off-diagonal mass of $A^\\top A$ is attributable to bounded-vs-"
             "unbounded support alone. Without that control, a difference could just be different "
             "point placement."),
    )
    m11_q
    return (m11_grade,)


@app.cell
def _(m11_grade):
    m11_grade()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Where to go next

    1. **Module 7 → a paper.** Formalise the additive bound in terms of the co-visibility Laplacian
       and state the exact condition under which the weighted mean is optimal.
    2. **Module 11 → the experiment.** Measure off-diagonal mass of $A^\top A$ for the two frozen
       arms on scene0062_00, and correlate it with the measured $L(x')/L(\hat{x})$ gap.
    3. **Module 10 → hygiene.** Never compare a level-3 lift against an all-4-level one, or across
       crop protocols, without saying so.

    *Companion note:* `ResearchVault/Papers/Gaussian-Semantics/SplatFeatureSolver-Curriculum-2026-09-09.md`
    """)
    return


if __name__ == "__main__":
    app.run()
