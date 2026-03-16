/**
 * EPOCH ZERO — Three.js 3D Globe Application
 * Real-time satellite orbit visualization with PINN-corrected trajectories
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// =========================================================================
// GLOBALS
// =========================================================================
let scene, camera, renderer, labelRenderer, controls;
let earthMesh, atmosphereMesh, starField;
let satelliteGroup, orbitGroup, sgp4Group, conjunctionGroup, labelGroup, dynamicLineGroup, debrisCloudGroup;
let scanData = null;
let animFrame = 0;
let animPlaying = false;
let animSpeed = 3;
let catalog = [];
let defaultIds = [];
let trackedSat = null;       // currently tracked satellite mesh
let raycaster, mouse;        // for click detection

const EARTH_RADIUS = 1.0;  // normalized
const SCALE_FACTOR = 1.0;  // trajectory data already normalized by R_EARTH in server

// =========================================================================
// INITIALIZATION
// =========================================================================
async function init() {
    // Set default date to tomorrow
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const targetDate = document.getElementById('target-date');
    if (targetDate) targetDate.value = tomorrow.toISOString().split('T')[0];

    // Slider label
    const propWindow = document.getElementById('prop-window');
    const propLabel = document.getElementById('prop-label');
    if (propWindow) {
        propWindow.addEventListener('input', (e) => {
            if (propLabel) propLabel.textContent = e.target.value + ' min';
        });
    }
    
    const speedSlider = document.getElementById('speed-slider');
    const speedLabel = document.getElementById('speed-label');
    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            animSpeed = parseInt(e.target.value);
            if (speedLabel) speedLabel.textContent = animSpeed + 'x';
        });
    }

    // Timeline scrubber
    const timelineScrubber = document.getElementById('timeline-scrubber');
    if (timelineScrubber) {
        timelineScrubber.addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            animFrame = val;
            updateTimelineLabel(val);
        });
    }

    // Toggle visibility
    const toggleOrbits = document.getElementById('toggle-orbits');
    if (toggleOrbits) {
        toggleOrbits.addEventListener('change', () => { if (orbitGroup) orbitGroup.visible = toggleOrbits.checked; });
    }
    
    const toggleConjunctions = document.getElementById('toggle-conjunctions');
    if (toggleConjunctions) {
        toggleConjunctions.addEventListener('change', () => { if (conjunctionGroup) conjunctionGroup.visible = toggleConjunctions.checked; });
    }
    
    const toggleStars = document.getElementById('toggle-stars');
    if (toggleStars) {
        toggleStars.addEventListener('change', () => { if (starField) starField.visible = toggleStars.checked; });
    }
    
    const toggleAtmosphere = document.getElementById('toggle-atmosphere');
    if (toggleAtmosphere) {
        toggleAtmosphere.addEventListener('change', () => { if (atmosphereMesh) atmosphereMesh.visible = toggleAtmosphere.checked; });
    }
    
    const toggleDebris = document.getElementById('toggle-debris');
    if (toggleDebris) {
        toggleDebris.addEventListener('change', () => { if (debrisCloudGroup) debrisCloudGroup.visible = toggleDebris.checked; });
    }
    
    const toggleLabels = document.getElementById('toggle-labels');
    const labelsContainer = document.getElementById('labels-container');
    if (toggleLabels && labelsContainer) {
        toggleLabels.addEventListener('change', () => { 
            labelsContainer.style.display = toggleLabels.checked ? 'block' : 'none'; 
        });
    }

    // Fetch catalog
    try {
        const res = await fetch('/api/info');
        const info = await res.json();
        catalog = info.catalog;
        defaultIds = info.defaults;
        
        const pinnLabel = document.getElementById('pinn-label');
        if (pinnLabel) pinnLabel.textContent = info.model_loaded ? 'PINN ONLINE' : 'PINN OFFLINE';
        
        const pinnDot = document.getElementById('pinn-dot');
        if (pinnDot && !info.model_loaded) pinnDot.classList.replace('dot-blue', 'dot-red');
        
        const footerModel = document.getElementById('footer-model');
        if (footerModel) footerModel.style.color = info.model_loaded ? 'var(--accent)' : 'var(--danger)';
        
        buildFleetList();
    } catch (e) {
        addLog('ERR', 'Failed to load catalog: ' + e.message);
    }

    // Clock with IST (Indian Standard Time UTC+5:30)
    function getISTTime() {
        const now = new Date();
        const istTime = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
        const year = istTime.getUTCFullYear();
        const month = String(istTime.getUTCMonth() + 1).padStart(2, '0');
        const day = String(istTime.getUTCDate()).padStart(2, '0');
        const hours = String(istTime.getUTCHours()).padStart(2, '0');
        const minutes = String(istTime.getUTCMinutes()).padStart(2, '0');
        const seconds = String(istTime.getUTCSeconds()).padStart(2, '0');
        return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
    }
    setInterval(() => {
        const clockPill = document.getElementById('clock-pill');
        if (clockPill) clockPill.textContent = '🕐 ' + getISTTime() + ' IST';
    }, 1000);

    initThreeJS();
    animate();
}

function buildFleetList() {
    const container = document.getElementById('fleet-list');
    if (!container) return;
    container.innerHTML = '';
    catalog.forEach(obj => {
        const tag = obj.type === 'PAYLOAD' ? 'SAT' : 'DEB';
        const tagCls = obj.type === 'PAYLOAD' ? 'tag-sat' : 'tag-deb';
        const checked = defaultIds.includes(obj.id) ? 'checked' : '';
        // Fallback: use name or ID if short is missing
        const displayName = obj.short || obj.name || obj.id || 'UNKNOWN';
        const displayId = obj.id || '?';
        container.innerHTML += `
            <label class="fleet-item" title="${obj.name || 'Unknown'} [${displayId}]">
                <input type="checkbox" value="${obj.id}" ${checked} class="fleet-cb">
                <span class="obj-dot" style="background:${obj.color || '#ffffff'}"></span>
                <span class="obj-label">${displayName}</span>
                <span class="obj-id" style="font-size: 11px; color: #667788; margin-left: auto;">[${displayId}]</span>
                <span class="obj-tag ${tagCls}">${tag}</span>
            </label>`;
    });
}

function getSelectedIds() {
    return Array.from(document.querySelectorAll('.fleet-cb:checked')).map(cb => cb.value);
}

// Expose to window for onclick handlers
window.selectAll = () => { document.querySelectorAll('.fleet-cb').forEach(cb => cb.checked = true); };
window.selectSats = () => { catalog.forEach((obj, i) => { document.querySelectorAll('.fleet-cb')[i].checked = obj.type === 'PAYLOAD'; }); };
window.selectDebris = () => { catalog.forEach((obj, i) => { document.querySelectorAll('.fleet-cb')[i].checked = obj.type === 'DEBRIS'; }); };
window.selectNone = () => { document.querySelectorAll('.fleet-cb').forEach(cb => cb.checked = false); };

// =========================================================================
// THREE.JS SETUP
// =========================================================================
function initThreeJS() {
    const container = document.getElementById('globe-container');
    if (!container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060810);

    // Camera
    camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
    camera.position.set(0, 0.8, 3.2);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);

    // CSS2D Renderer for HTML Labels
    labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(w, h);
    labelRenderer.domElement.style.position = 'absolute';
    labelRenderer.domElement.style.top = '0px';
    labelRenderer.domElement.style.pointerEvents = 'none';
    const labelsContainerAppend = document.getElementById('labels-container');
    if (labelsContainerAppend) labelsContainerAppend.appendChild(labelRenderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.5;
    controls.maxDistance = 15;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;

    // Raycaster for satellite click detection
    raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.05 };
    mouse = new THREE.Vector2();
    renderer.domElement.addEventListener('click', onSatelliteClick);
    renderer.domElement.addEventListener('dblclick', () => { untrackSatellite(); });

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x334466, 0.6);
    scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.8);
    sunLight.position.set(5, 3, 4);
    scene.add(sunLight);

    const fillLight = new THREE.DirectionalLight(0x4488cc, 0.3);
    fillLight.position.set(-3, -1, -2);
    scene.add(fillLight);

    // Create scene elements
    createStarField();
    createEarth();
    createAtmosphere();

    // Groups for dynamic elements
    satelliteGroup = new THREE.Group();
    orbitGroup = new THREE.Group();
    sgp4Group = new THREE.Group();
    conjunctionGroup = new THREE.Group();
    labelGroup = new THREE.Group();
    dynamicLineGroup = new THREE.Group();
    debrisCloudGroup = new THREE.Group();
    scene.add(satelliteGroup);
    scene.add(orbitGroup);
    scene.add(sgp4Group);
    scene.add(conjunctionGroup);
    scene.add(labelGroup);
    scene.add(dynamicLineGroup);
    scene.add(debrisCloudGroup);
    
    // Create initial debris cloud
    createDebrisCloud();

    // Resize handler
    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
        labelRenderer.setSize(w, h);
    });
}

function createStarField() {
    const starsGeo = new THREE.BufferGeometry();
    const starCount = 3000;
    const positions = new Float32Array(starCount * 3);
    const sizes = new Float32Array(starCount);

    for (let i = 0; i < starCount; i++) {
        const r = 30 + Math.random() * 20;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
        sizes[i] = 0.3 + Math.random() * 1.2;
    }

    starsGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    starsGeo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    const starsMat = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.08,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
    });

    starField = new THREE.Points(starsGeo, starsMat);
    scene.add(starField);
}

function createEarth() {
    // Procedural Earth texture
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Ocean base
    const oceanGrad = ctx.createRadialGradient(512, 256, 0, 512, 256, 512);
    oceanGrad.addColorStop(0, '#1a3a5c');
    oceanGrad.addColorStop(0.5, '#0f2847');
    oceanGrad.addColorStop(1, '#0a1e3a');
    ctx.fillStyle = oceanGrad;
    ctx.fillRect(0, 0, 1024, 512);

    // Continent shapes (simplified procedural)
    ctx.fillStyle = '#1a4a3a';
    const continents = [
        // North America
        [[200,100],[280,90],[320,130],[310,180],[280,200],[240,220],[200,200],[180,160]],
        // South America
        [[260,240],[290,250],[300,300],[290,360],[260,380],[240,340],[250,280]],
        // Europe+Africa
        [[480,100],[520,90],[540,120],[560,140],[550,180],[530,200],[560,260],[540,340],[510,360],[490,300],[470,240],[460,180],[470,140]],
        // Asia
        [[560,80],[650,70],[720,90],[760,110],[740,150],[700,160],[680,140],[640,130],[600,140],[580,120]],
        // Australia
        [[720,280],[770,270],[790,290],[780,320],[750,330],[720,310]],
        // Antarctica
        [[300,440],[400,450],[500,445],[600,450],[700,440],[650,480],[350,480]],
    ];

    continents.forEach(pts => {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
            const prev = pts[i - 1];
            const curr = pts[i];
            const cpx = (prev[0] + curr[0]) / 2 + (Math.random() - 0.5) * 20;
            const cpy = (prev[1] + curr[1]) / 2 + (Math.random() - 0.5) * 15;
            ctx.quadraticCurveTo(cpx, cpy, curr[0], curr[1]);
        }
        ctx.closePath();
        ctx.fillStyle = 'rgba(25, 80, 60, 0.7)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(40, 120, 90, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });

    // Grid lines
    ctx.strokeStyle = 'rgba(60, 130, 200, 0.08)';
    ctx.lineWidth = 0.5;
    for (let lat = 0; lat <= 512; lat += 512 / 18) {
        ctx.beginPath(); ctx.moveTo(0, lat); ctx.lineTo(1024, lat); ctx.stroke();
    }
    for (let lon = 0; lon <= 1024; lon += 1024 / 36) {
        ctx.beginPath(); ctx.moveTo(lon, 0); ctx.lineTo(lon, 512); ctx.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;

    const earthGeo = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
    const earthMat = new THREE.MeshPhongMaterial({
        map: texture,
        specular: new THREE.Color(0x222244),
        shininess: 25,
        emissive: new THREE.Color(0x050a15),
        emissiveIntensity: 0.3,
    });

    earthMesh = new THREE.Mesh(earthGeo, earthMat);
    scene.add(earthMesh);
}

function createAtmosphere() {
    // Glow atmosphere using a slightly larger sphere with backside rendering
    const atmoGeo = new THREE.SphereGeometry(EARTH_RADIUS * 1.02, 64, 64);
    const atmoMat = new THREE.ShaderMaterial({
        vertexShader: `
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {
                vNormal = normalize(normalMatrix * normal);
                vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `,
        fragmentShader: `
            varying vec3 vNormal;
            varying vec3 vPosition;
            void main() {
                float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 3.0);
                gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity * 0.7;
            }
        `,
        blending: THREE.AdditiveBlending,
        side: THREE.BackSide,
        transparent: true,
    });

    atmosphereMesh = new THREE.Mesh(atmoGeo, atmoMat);
    atmosphereMesh.scale.setScalar(1.15);
    scene.add(atmosphereMesh);
}

function createDebrisCloud() {
    // Create animated debris cloud around Earth showing LEO threat
    while (debrisCloudGroup.children.length > 0) {
        debrisCloudGroup.removeChild(debrisCloudGroup.children[0]);
    }
    const debrisCount = 800;
    const debrisColors = [0xff5500, 0xff6600, 0xff7700, 0xff8800, 0xff3300, 0xff4400];
    const minAltitude = 1.06;  // ~400 km
    const maxAltitude = 1.31;  // ~2000 km
    for (let i = 0; i < debrisCount; i++) {
        const altitude = minAltitude + Math.random() * (maxAltitude - minAltitude);
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.random() * Math.PI;
        const x = altitude * Math.sin(phi) * Math.cos(theta);
        const y = altitude * Math.sin(phi) * Math.sin(theta);
        const z = altitude * Math.cos(phi);
        const size = 0.003 + Math.random() * 0.008;
        const debrisGeo = new THREE.SphereGeometry(size, 4, 4);
        const debrisColor = debrisColors[Math.floor(Math.random() * debrisColors.length)];
        const debrisMat = new THREE.MeshPhongMaterial({
            color: debrisColor, emissive: debrisColor, emissiveIntensity: 0.5,
            transparent: true, opacity: 0.6
        });
        const debris = new THREE.Mesh(debrisGeo, debrisMat);
        debris.position.set(x, y, z);
        debris.userData = {orbitSpeed: 0.001 + Math.random() * 0.002, theta, phi, altitude};
        debrisCloudGroup.add(debris);
    }
    debrisCloudGroup.visible = true;
}

// =========================================================================
// SATELLITE VISUALIZATION
// =========================================================================
function clearScene() {
    [satelliteGroup, orbitGroup, sgp4Group, conjunctionGroup, dynamicLineGroup].forEach(group => {
        if (!group) return;
        while (group.children.length > 0) {
            const child = group.children[0];
            if (child.isCSS2DObject && child.element) child.element.remove();
            if (child.children) child.children.forEach(c => {
                if (c.isCSS2DObject && c.element) c.element.remove();
            });
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                else child.material.dispose();
            }
            group.remove(child);
        }
    });
    document.querySelectorAll('.sat-label-3d, .dist-label-3d').forEach(el => el.remove());
}

function populateScene(data) {
    clearScene();
    scanData = data;
    animFrame = 0;

    const objects = data.objects;
    const objIds = Object.keys(objects);

    objIds.forEach(oid => {
        const obj = objects[oid];
        const color = new THREE.Color(obj.color);

        // Satellite marker
        const isPayload = obj.type === 'PAYLOAD';
        const markerGeo = isPayload ? new THREE.OctahedronGeometry(0.02, 0) : new THREE.SphereGeometry(0.015, 8, 8);
        const markerMat = new THREE.MeshPhongMaterial({ color: color, emissive: color, emissiveIntensity: 0.6 });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.userData = { oid, trajectory: obj.trajectory_pinn_eci, name: obj.short, color: obj.color };

        if (obj.trajectory_pinn_eci && obj.trajectory_pinn_eci.length > 0) {
            const p = obj.trajectory_pinn_eci[0];
            marker.position.set(p[0], p[2], -p[1]);
        }
        satelliteGroup.add(marker);

        // 3D Label
        const labelDiv = document.createElement('div');
        labelDiv.className = 'sat-label-3d';
        labelDiv.innerHTML = `<span>${obj.short}</span><div class="sat-details">ALT: ${obj.altitude.toFixed(0)} km<br>SPD: ${obj.speed.toFixed(3)} km/s</div>`;
        const label = new CSS2DObject(labelDiv);
        marker.add(label);

        // Trajectory Trails (Rendered as rails via CatmullRomCurve3 for smoothness)
        if (obj.trajectory_pinn_eci && obj.trajectory_pinn_eci.length > 1) {
            // PINN Corrected (Active Path)
            const points = obj.trajectory_pinn_eci.map(p => new THREE.Vector3(p[0], p[2], -p[1]));
            const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
            // Oversample by 4x for smooth rail rendering (min 2000 points)
            const numRenderPoints = Math.max(points.length * 4, 2000);
            const renderPoints = curve.getPoints(numRenderPoints);
            const lineMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.4 });
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(renderPoints);
            const orbitLine = new THREE.Line(lineGeometry, lineMat);
            orbitLine.userData = { 
                origLength: points.length, 
                renderLength: renderPoints.length,
                renderPointsPerOrbit: Math.floor(480 * (renderPoints.length / points.length))
            };
            orbitGroup.add(orbitLine);

            // SGP4 Physics (Phantom Path)
            if (obj.trajectory_sgp4_eci) {
                const pSgp4 = obj.trajectory_sgp4_eci.map(p => new THREE.Vector3(p[0], p[2], -p[1]));
                const curveSgp4 = new THREE.CatmullRomCurve3(pSgp4, false, 'catmullrom', 0.5);
                const numSgp4RenderPoints = Math.max(pSgp4.length * 4, 2000);
                const renderSgp4 = curveSgp4.getPoints(numSgp4RenderPoints);
                const sgp4Mat = new THREE.LineDashedMaterial({ color: 0x888888, dashSize: 0.02, gapSize: 0.01, transparent: true, opacity: 0.2 });
                const sgp4Geometry = new THREE.BufferGeometry().setFromPoints(renderSgp4);
                const sgp4Line = new THREE.Line(sgp4Geometry, sgp4Mat);
                sgp4Line.computeLineDistances();
                sgp4Line.userData = { 
                    origLength: pSgp4.length, 
                    renderLength: renderSgp4.length,
                    renderPointsPerOrbit: Math.floor(480 * (renderSgp4.length / pSgp4.length))
                };
                sgp4Group.add(sgp4Line);
            }
        }
    });

    const avgDrift = objIds.length ? (Object.values(objects).reduce((s, o) => s + o.dr, 0) / objIds.length) : 0;
    const diagDrift = document.getElementById('diag-drift');
    if (diagDrift) diagDrift.textContent = avgDrift.toFixed(2) + ' km';
    const hudObjects = document.getElementById('hud-objects');
    if (hudObjects) hudObjects.textContent = objIds.length;
    
    animPlaying = true;
    const btnPlay = document.getElementById('btn-play');
    if (btnPlay) btnPlay.classList.add('active');

    // Setup timeline scrubber range
    const timelineScrubber = document.getElementById('timeline-scrubber');
    if (timelineScrubber) {
        const firstObj = Object.values(data.objects)[0];
        const maxFrame = (firstObj && firstObj.trajectory_pinn_eci) ? firstObj.trajectory_pinn_eci.length - 1 : 600;
        timelineScrubber.max = maxFrame;
        timelineScrubber.value = 0;
    }
    updateTimelineLabel(0);
}

window.addCustomSatellite = async function() {
    const customNorad = document.getElementById('custom-norad');
    if (!customNorad) return;
    const id = customNorad.value.trim();
    if (!id) return;
    
    const btn = document.getElementById('btn-add-sat');
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = '🛰️ Tracking...';
    
    try {
        const res = await fetch('/api/add-sat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id })
        });
        const data = await res.json();
        if (data.success) {
            addLog('OK', `Target ${data.name} [NORAD ${data.id}] verified and locked.`);
            const resInfo = await fetch('/api/info');
            catalog = (await resInfo.json()).catalog;
            buildFleetList(); // Refresh the list
            if (customNorad) customNorad.value = '';
        } else {
            addLog('ERR', data.error || 'Server error - check if backend is running.');
        }
    } catch (e) {
        addLog('ERR', 'Target lock failed: ' + e.message);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Lock Target';
        }
    }
};

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    if (earthMesh) earthMesh.rotation.y += 0.0002;
    
    // Animate debris cloud
    if (debrisCloudGroup) {
        debrisCloudGroup.children.forEach(debris => {
            if (debris.userData) {
                debris.userData.theta += debris.userData.orbitSpeed;
                const alt = debris.userData.altitude;
                const theta = debris.userData.theta;
                const phi = debris.userData.phi;
                debris.position.x = alt * Math.sin(phi) * Math.cos(theta);
                debris.position.z = alt * Math.sin(phi) * Math.sin(theta);
                debris.position.y = alt * Math.cos(phi);
            }
        });
    }
    
    const toggleSgp4 = document.getElementById('toggle-sgp4');
    if (sgp4Group && toggleSgp4) sgp4Group.visible = toggleSgp4.checked;

    if (animPlaying && scanData && scanData.objects) {
        // Determine total frames from actual trajectory data
        const firstObj = Object.values(scanData.objects)[0];
        const totalFrames = (firstObj && firstObj.trajectory_pinn_eci) ? firstObj.trajectory_pinn_eci.length : (scanData.total_frames || 600);
        // Warp speed: multiplier directly controls frame step
        // Reduced from 0.5 to 0.02 to make 1x look close to live speed
        animFrame = (animFrame + animSpeed * 0.02) % totalFrames;
        const f = Math.floor(animFrame);
        
        // Update HUD frame
        const orbitPct = ((f / totalFrames) * 100).toFixed(0);
        const hudFrame = document.getElementById('hud-frame');
        if (hudFrame) hudFrame.textContent = `${f + 1}/${totalFrames} (${orbitPct}%)`;
        
        // Update orbital epoch with full date+time from server target_epoch
        updateEpochDisplay(f, totalFrames);

        // Update timeline scrubber position
        const timelineScrubber = document.getElementById('timeline-scrubber');
        if (timelineScrubber) timelineScrubber.value = f;
        updateTimelineLabel(f);

        satelliteGroup.children.forEach(child => {
            if (child.userData.trajectory) {
                const p = child.userData.trajectory[Math.min(f, child.userData.trajectory.length - 1)];
                child.position.set(p[0], p[2], -p[1]);
            }
        });

        // Update dynamic draw range for orbit trails to prevent spaghetti overlapping
        const updateDrawRange = (line) => {
            if (line.userData && line.userData.origLength) {
                const renderRatio = line.userData.renderLength / line.userData.origLength;
                const currentRenderFrame = Math.floor(animFrame * renderRatio);
                // Draw roughly half an orbit behind and half ahead
                const halfOrbit = Math.floor(line.userData.renderPointsPerOrbit / 2);
                const start = Math.max(0, currentRenderFrame - halfOrbit);
                const targetEnd = currentRenderFrame + halfOrbit;
                const count = Math.min(targetEnd - start, line.userData.renderLength - start);
                line.geometry.setDrawRange(start, count);
            }
        };
        orbitGroup.children.forEach(updateDrawRange);
        sgp4Group.children.forEach(updateDrawRange);

        // Camera tracking: smoothly follow the tracked satellite
        if (trackedSat && trackedSat.userData.trajectory) {
            const satPos = trackedSat.position.clone();
            const offset = satPos.clone().normalize().multiplyScalar(0.5);
            const targetCamPos = satPos.clone().add(offset);
            camera.position.lerp(targetCamPos, 0.05);
            controls.target.lerp(satPos, 0.08);
        }

        // Dynamic distance lines
        while (dynamicLineGroup.children.length > 0) {
            const child = dynamicLineGroup.children[0];
            if (child.children) child.children.forEach(c => { if(c.isCSS2DObject && c.element) c.element.remove(); });
            dynamicLineGroup.remove(child);
        }

        if (scanData.pairs) {
            const toggleConjunctions = document.getElementById('toggle-conjunctions');
            const showConjunctions = toggleConjunctions ? toggleConjunctions.checked : true;
            if (showConjunctions) {
                scanData.pairs.forEach(pair => {
                    const markerA = satelliteGroup.children.find(c => c.userData.oid === pair.a);
                    const markerB = satelliteGroup.children.find(c => c.userData.oid === pair.b);
                    if (markerA && markerB) {
                        const distKm = markerA.position.distanceTo(markerB.position) * 6371;
                        if (distKm < 2000) {
                            const lineMat = new THREE.LineDashedMaterial({ color: distKm < 500 ? 0xef4444 : 0xf59e0b, dashSize: 0.05, gapSize: 0.03, transparent: true, opacity: 0.8 });
                            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([markerA.position, markerB.position]), lineMat);
                            line.computeLineDistances();
                            
                            const labelDiv = document.createElement('div');
                            labelDiv.className = 'dist-label-3d' + (distKm < 500 ? ' crit' : '');
                            labelDiv.textContent = distKm.toFixed(1) + ' km';
                            const label = new CSS2DObject(labelDiv);
                            label.position.copy(new THREE.Vector3().addVectors(markerA.position, markerB.position).multiplyScalar(0.5));
                            line.add(label);
                            dynamicLineGroup.add(line);
                        }
                    }
                });
            }
        }
    }
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}

// =========================================================================
// EPOCH & TIMELINE HELPERS
// =========================================================================
function updateEpochDisplay(frame, totalFrames) {
    const hudTime = document.getElementById('hud-time');
    if (!hudTime) return;

    if (scanData && scanData.start_epoch) {
        // Simulation starts at NOW (start_epoch), covers prop_window minutes
        const startDate = new Date(scanData.start_epoch);
        const propWin = scanData.prop_window || 100;
        const msPerFrame = (propWin * 60 * 1000) / totalFrames;
        const simDate = new Date(startDate.getTime() + frame * msPerFrame);
        const yr = simDate.getUTCFullYear();
        const mo = String(simDate.getUTCMonth() + 1).padStart(2, '0');
        const dy = String(simDate.getUTCDate()).padStart(2, '0');
        const hh = String(simDate.getUTCHours()).padStart(2, '0');
        const mm = String(simDate.getUTCMinutes()).padStart(2, '0');
        const ss = String(simDate.getUTCSeconds()).padStart(2, '0');
        hudTime.textContent = `${yr}-${mo}-${dy} ${hh}:${mm}:${ss} UTC`;
    } else {
        const now = new Date();
        hudTime.textContent = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }
}

function updateTimelineLabel(frame) {
    const label = document.getElementById('timeline-label');
    if (!label || !scanData) return;
    const firstObj = Object.values(scanData.objects || {})[0];
    const totalFrames = (firstObj && firstObj.trajectory_pinn_eci) ? firstObj.trajectory_pinn_eci.length : 600;
    const propWin = scanData.prop_window || 100;
    const minutesElapsed = ((frame / totalFrames) * propWin).toFixed(1);
    label.textContent = `T+${minutesElapsed} min`;
}

// =========================================================================
// SATELLITE CLICK TRACKING
// =========================================================================
function onSatelliteClick(event) {
    if (!satelliteGroup || satelliteGroup.children.length === 0) return;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(satelliteGroup.children, false);
    if (intersects.length > 0) {
        const sat = intersects[0].object;
        trackSatellite(sat);
    }
}

function trackSatellite(sat) {
    // Unhighlight previous
    if (trackedSat) {
        trackedSat.material.emissiveIntensity = 0.6;
        trackedSat.scale.set(1, 1, 1);
    }
    trackedSat = sat;
    trackedSat.material.emissiveIntensity = 1.2;
    trackedSat.scale.set(1.8, 1.8, 1.8);
    controls.autoRotate = false;
    addLog('TRK', `Tracking: ${sat.userData.name || sat.userData.oid}`);
    // Update tracking HUD
    const trackLabel = document.getElementById('track-label');
    if (trackLabel) trackLabel.textContent = sat.userData.name || sat.userData.oid;
    const trackPanel = document.getElementById('track-panel');
    if (trackPanel) trackPanel.style.display = 'block';
}

window.untrackSatellite = function() {
    if (trackedSat) {
        trackedSat.material.emissiveIntensity = 0.6;
        trackedSat.scale.set(1, 1, 1);
        addLog('TRK', `Untracked: ${trackedSat.userData.name || trackedSat.userData.oid}`);
        trackedSat = null;
    }
    controls.autoRotate = true;
    const trackPanel = document.getElementById('track-panel');
    if (trackPanel) trackPanel.style.display = 'none';
};

// =========================================================================
// API CALLS
// =========================================================================
window.runScan = async function () {
    const ids = getSelectedIds();
    if (ids.length < 2) {
        addLog('ERR', 'Select at least 2 objects for batch scan.');
        return;
    }

    const targetDateEl = document.getElementById('target-date');
    const targetTimeEl = document.getElementById('target-time');
    const propWindowEl = document.getElementById('prop-window');
    
    const targetDate = targetDateEl ? targetDateEl.value : '';
    const targetTime = targetTimeEl ? targetTimeEl.value : '12:00:00';
    const propWindow = propWindowEl ? parseInt(propWindowEl.value) : 100;

    addLog('CMD', `Batch scan: ${ids.length} objects, ${ids.length * (ids.length - 1) / 2} pairs`);

    const btnScan = document.getElementById('btn-scan');
    if (btnScan) {
        btnScan.disabled = true;
        btnScan.textContent = '⏳ Scanning...';
    }

    try {
        const res = await fetch('/api/scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids, target: `${targetDate} ${targetTime}`, prop_window: propWindow }),
        });

        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            const text = await res.text();
            addLog('ERR', 'Server Error: Returned HTML instead of JSON. Check backend.');
            console.error("Non-JSON response:", text);
            if (btnScan) {
                btnScan.disabled = false;
                btnScan.textContent = '🔍 Batch Conjunction Scan';
            }
            return;
        }

        const data = await res.json();

        if (!res.ok) {
            addLog('ERR', data.error || 'Scan failed');
            if (data.details) data.details.forEach(d => addLog('ERR', d));
            if (btnScan) {
                btnScan.disabled = false;
                btnScan.textContent = '🔍 Batch Conjunction Scan';
            }
            return;
        }

        if (data.errors) data.errors.forEach(e => addLog('ERR', e));
        addLog('OK', `Batch complete: ${data.n_objects} objects, ${data.n_pairs} pairs analyzed`);
        if (data.pairs && data.pairs.length > 0) {
            addLog('TCA', `Closest: ${data.closest_pair} @ ${data.min_miss} km at ${data.pairs[0].tca}`);
        } else {
            addLog('TCA', `Closest: ${data.closest_pair} @ ${data.min_miss} km`);
        }

        populateScene(data);
        updateRightPanel(data);

    } catch (e) {
        addLog('ERR', 'Network error: ' + e.message);
    } finally {
        if (btnScan) {
            btnScan.disabled = false;
            btnScan.textContent = '🔍 Batch Conjunction Scan';
        }
    }
};

window.fetchTLEs = async function () {
    const ids = getSelectedIds();
    addLog('CMD', `Fetching TLEs for ${ids.length || 'all'} objects...`);
    try {
        const res = await fetch('/api/fetch-tles', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids: ids.length ? ids : undefined }),
        });
        const data = await res.json();
        if (data.error) {
            addLog('ERR', data.error);
        } else {
            addLog('OK', `Fetched ${data.ok} live TLEs (${data.fail} failed)`);
            const tleSource = document.getElementById('tle-source');
            if (tleSource) {
                tleSource.textContent = data.source;
                tleSource.style.color = 'var(--success)';
            }
        }
    } catch (e) {
        addLog('ERR', 'Fetch failed: ' + e.message);
    }
};

// =========================================================================
// RIGHT PANEL UPDATES
// =========================================================================
function updateRightPanel(data) {
    // Fleet Status
    const nCrit = data.n_critical;
    let badgeCls, badgeTxt;
    if (nCrit > 0) { badgeCls = 'badge-critical'; badgeTxt = 'CRITICAL'; }
    else if (data.pairs.some(p => p.threat === 'WARNING')) { badgeCls = 'badge-warn'; badgeTxt = 'WARNING'; }
    else { badgeCls = 'badge-low'; badgeTxt = 'LOW'; }

    const fleetStatusBody = document.getElementById('fleet-status-body');
    if (fleetStatusBody) {
        fleetStatusBody.innerHTML = `
            <div class="stat-grid">
                <div class="stat-item"><span class="stat-label">OBJECTS:</span><span class="stat-value" style="color:var(--accent-light);">${data.n_objects}</span></div>
                <div class="stat-item"><span class="stat-label">PAIRS:</span><span class="stat-value" style="color:var(--purple);">${data.n_pairs}</span></div>
                <div class="stat-item"><span class="stat-label">ALERTS:</span><span class="stat-value" style="color:var(--danger);">${nCrit}</span></div>
                <div class="stat-item"><span class="stat-label">MIN DIST:</span><span class="stat-value" style="color:var(--warning);">${data.min_miss} km</span></div>
            </div>
            <div class="threat-level">
                <span class="threat-label">Fleet Threat</span>
                <span class="threat-badge ${badgeCls}">${badgeTxt}</span>
            </div>`;
    }

    // Risk Table
    const colorMap = { CRITICAL: 'var(--danger)', HIGH: 'var(--danger)', WARNING: 'var(--warning)', LOW: 'var(--success)' };
    const bgMap = { CRITICAL: 'threat-bg-critical', HIGH: 'threat-bg-high', WARNING: 'threat-bg-warn', LOW: 'threat-bg-low' };
    
    let tableHtml = `<table class="risk-table"><thead><tr>
        <th onclick="sortTable(0)">Pair <span class="sort-arrow">↕</span></th>
        <th>TCA (Time)</th>
        <th onclick="sortTable(1)" class="sorted">Min Distance <span class="sort-arrow">↑</span></th>
        <th>Threat</th></tr></thead><tbody>`;
    data.pairs.forEach(p => {
        const bgRowCls = bgMap[p.threat] || 'threat-bg-low';
        tableHtml += `<tr class="${bgRowCls}">
            <td>${p.name_a} ↔ ${p.name_b}</td>
            <td style="opacity:0.8; font-size:11px;">${p.tca}</td>
            <td style="color:${colorMap[p.threat]}">${p.miss_dist} km</td>
            <td><span class="threat-cell">${p.threat}</span></td></tr>`;
    });
    tableHtml += '</tbody></table>';
    const riskTableBody = document.getElementById('risk-table-body');
    if (riskTableBody) riskTableBody.innerHTML = tableHtml;

    // Top Threats
    let threatsHtml = '';
    data.pairs.slice(0, 7).forEach((p, i) => {
        const c = colorMap[p.threat];
        const cls = p.threat === 'CRITICAL' ? 'badge-critical' : p.threat === 'HIGH' ? 'badge-high' : p.threat === 'WARNING' ? 'badge-warn' : 'badge-low';
        threatsHtml += `<div class="threat-item">
            <span class="threat-rank" style="color:${c}">#${i + 1}</span>
            <span class="threat-pair"><span class="obj-a">${p.name_a}</span><span class="sep">↔</span><span class="obj-b">${p.name_b}</span></span>
            <span class="threat-dist" style="color:${c}">${p.miss_dist} km</span>
            <span class="threat-badge ${cls}">${p.threat}</span>
        </div>`;
    });
    const threatsBody = document.getElementById('threats-body');
    if (threatsBody) threatsBody.innerHTML = threatsHtml || '<div class="placeholder">No pairs.</div>';

    // Telemetry
    let telemHtml = '<table class="telem-table"><thead><tr><th>Object</th><th>ALT (km)</th><th>SPD (km/s)</th><th>Δr (km)</th><th>TLE Epoch</th></tr></thead><tbody>';
    let hasData = false;
    Object.values(data.objects).forEach(obj => {
        hasData = true;
        const tagCls = obj.type === 'PAYLOAD' ? 'tag-sat' : 'tag-deb';
        const tag = obj.type === 'PAYLOAD' ? 'SAT' : 'DEB';
        const tleEpoch = obj.tle_epoch || '---';
        telemHtml += `<tr>
            <td>
                <div class="telem-name-cell">
                    <span class="telem-dot" style="background:${obj.color}; width:8px; height:8px; border-radius:50%; display:inline-block;"></span>
                    <span style="color:${obj.color}">${obj.short}</span>
                    <span class="telem-type ${tagCls}">${tag}</span>
                </div>
            </td>
            <td>${obj.altitude.toFixed(0)}</td>
            <td>${obj.speed.toFixed(3)}</td>
            <td>${obj.dr.toFixed(2)}</td>
            <td class="tle-epoch-cell">${tleEpoch}</td>
        </tr>`;
    });
    telemHtml += '</tbody></table>';
    const telemetryBody = document.getElementById('telemetry-body');
    if (telemetryBody) telemetryBody.innerHTML = hasData ? telemHtml : '<div class="placeholder">No data.</div>';
}

// Table sorting
let sortCol = 1, sortAsc = true;
window.sortTable = function (col) {
    if (sortCol === col) sortAsc = !sortAsc;
    else { sortCol = col; sortAsc = true; }
    if (!scanData) return;
    const pairs = [...scanData.pairs];
    pairs.sort((a, b) => {
        let va, vb;
        if (col === 0) { va = a.name_a + a.name_b; vb = b.name_a + b.name_b; }
        else { va = a.miss_dist; vb = b.miss_dist; }
        if (va < vb) return sortAsc ? -1 : 1;
        if (va > vb) return sortAsc ? 1 : -1;
        return 0;
    });
    scanData.pairs = pairs;
    updateRightPanel(scanData);
};

// =========================================================================
// CONTROLS
// =========================================================================
window.togglePlay = function () {
    animPlaying = !animPlaying;
    const btnPlay = document.getElementById('btn-play');
    if (btnPlay) {
        btnPlay.classList.toggle('active', animPlaying);
        btnPlay.textContent = animPlaying ? '▶ Playing' : '▶ Play';
    }
};

window.pauseAnim = function () {
    animPlaying = false;
    const btnPlay = document.getElementById('btn-play');
    if (btnPlay) {
        btnPlay.classList.remove('active');
        btnPlay.textContent = '▶ Play';
    }
};

window.resetCamera = function () {
    if (trackedSat) untrackSatellite();
    camera.position.set(0, 0.8, 3.2);
    controls.target.set(0, 0, 0);
    controls.update();
};

// =========================================================================
// EVENT LOG
// =========================================================================
function addLog(tag, msg) {
    const logEl = document.getElementById('event-log');
    if (!logEl) return;
    const tagClass = { SYS: 'log-sys', CMD: 'log-cmd', OK: 'log-ok', ERR: 'log-err', TCA: 'log-tca', TRK: 'log-tca', NET: 'log-cmd' }[tag] || 'log-sys';
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-tag ${tagClass}">${tag}</span><span class="log-msg">${msg}</span>`;
    logEl.insertBefore(entry, logEl.firstChild);
    // Keep only last 30
    while (logEl.children.length > 30) logEl.removeChild(logEl.lastChild);
}

// =========================================================================
// START
// =========================================================================
init();
