// ==================== VARIABLES GLOBALES ====================
let basicData = null;
let riskData = null;
let mlRFData = null;
let mlXBData = null;
const sections = ['inicio', 'datos', 'analisis', 'factores', 'ml'];
const dots = document.querySelectorAll('.nav-dot');
const fipsToStateName = { '01': 'Alabama', '02': 'Alaska', '04': 'Arizona', '05': 'Arkansas', '06': 'California', '08': 'Colorado', '09': 'Connecticut', '10': 'Delaware', '11': 'District of Columbia', '12': 'Florida', '13': 'Georgia', '15': 'Hawaii', '16': 'Idaho', '17': 'Illinois', '18': 'Indiana', '19': 'Iowa', '20': 'Kansas', '21': 'Kentucky', '22': 'Louisiana', '23': 'Maine', '24': 'Maryland', '25': 'Massachusetts', '26': 'Michigan', '27': 'Minnesota', '28': 'Mississippi', '29': 'Missouri', '30': 'Montana', '31': 'Nebraska', '32': 'Nevada', '33': 'New Hampshire', '34': 'New Jersey', '35': 'New Mexico', '36': 'New York', '37': 'North Carolina', '38': 'North Dakota', '39': 'Ohio', '40': 'Oklahoma', '41': 'Oregon', '42': 'Pennsylvania', '44': 'Rhode Island', '45': 'South Carolina', '46': 'South Dakota', '47': 'Tennessee', '48': 'Texas', '49': 'Utah', '50': 'Vermont', '51': 'Virginia', '53': 'Washington', '54': 'West Virginia', '55': 'Wisconsin', '56': 'Wyoming' };



document.addEventListener('DOMContentLoaded', async function () {
    try {
        //Carga de datos
        const [basicRes, riskRes, mlRFRes, mlXBRes] = await Promise.all([
            fetch('data/03_basic_analysis.json'),
            fetch('data/05_advanced_risks.json'),
            fetch('data/06_ml_model.json'),
            fetch('data/08_ml_insights.json'),

        ]);
        basicData = await basicRes.json();
        riskData = await riskRes.json();
        mlRFData = await mlRFRes.json();
        mlXBData = await mlXBRes.json();

        initTabs();
        updateDatasetSection()
        await loadMap();
        updateBasicAnalysisSection();
        riskFactor();
        mlAnalisis()

    } catch (error) {
        console.error('Error fatal:', error);
        document.querySelector('#statsDatos').innerHTML = '<div class="stat-card" style="grid-column:1/-1; color:#ef4444;">❌ No se pudieron cargar los datos. Verifica que los archivos JSON estén en la ruta "data/".</div>';
    }
});


// ==================== EVENTOS TAB ====================
function initTabs() {
    document.querySelectorAll('.glass-card').forEach(card => {
        const btns = card.querySelectorAll('.tab-btn'), panes = card.querySelectorAll('.tab-pane');
        btns.forEach(btn => btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            btns.forEach(b => b.classList.remove('active')); btn.classList.add('active');
            panes.forEach(p => p.classList.remove('active-pane'));
            const target = card.querySelector(`#${tabId}`);
            if (target) target.classList.add('active-pane');
        }));
    });
}
window.addEventListener('scroll', () => {
    let current = '';
    for (let s of sections) { const el = document.getElementById(s); if (el) { const rect = el.getBoundingClientRect(); if (rect.top <= 200 && rect.bottom >= 200) current = s; } }
    dots.forEach(dot => { dot.classList.remove('active'); if (dot.getAttribute('data-section') === current) dot.classList.add('active'); });
});
dots.forEach(dot => dot.addEventListener('click', () => document.getElementById(dot.getAttribute('data-section')).scrollIntoView({ behavior: 'smooth' })));



// ==================== FUENTE DE DATOS ====================
function updateDatasetSection() {
    if (!basicData) return;
    const stats = basicData.estadisticas_generales;
    document.getElementById('total-records').textContent = stats.total_registros.toLocaleString();
    document.getElementById('total-columns').textContent = stats.total_columnas;
    document.getElementById('states-count').textContent = Object.keys(basicData.por_estado).length;
    document.getElementById('prevalence').textContent = stats.prevalencia_porcentaje + '%';
}


// ==================== MAPA ====================
async function loadMap() {
    if (!basicData) return;
    try {
        const us = await fetch('https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json').then(r => r.json());
        const states = topojson.feature(us, us.objects.states).features;
        const svg = d3.select('#map').append('svg').attr('width', '100%').attr('height', '100%').attr('viewBox', '0 0 975 610').attr('preserveAspectRatio', 'xMidYMid meet');
        const projection = d3.geoAlbersUsa().fitSize([975, 610], topojson.feature(us, us.objects.nation));
        const path = d3.geoPath().projection(projection);
        const prevalencias = Object.values(basicData.por_estado).map(v => parseFloat(v.prevalencia_porcentaje));
        const minPrev = Math.min(...prevalencias), maxPrev = Math.max(...prevalencias);
        const colorScale = d3.scaleLinear().domain([minPrev, maxPrev]).range(['#fecaca', '#991b1b']);
        const tooltip = d3.select('body').append('div').style('position', 'absolute').style('background', '#0f0b18e6').style('color', '#fff').style('padding', '8px 14px').style('border-radius', '20px').style('border-left', '4px solid #ef4444').style('font-size', '12px').style('backdrop-filter', 'blur(8px)').style('pointer-events', 'none').style('z-index', '2000').style('display', 'none');
        svg.selectAll('.state').data(states).enter().append('path').attr('class', 'state').attr('d', path).attr('fill', d => { const name = fipsToStateName[String(d.id).padStart(2, '0')]; if (basicData.por_estado[name]) return colorScale(parseFloat(basicData.por_estado[name].prevalencia_porcentaje)); return '#4a2e3a'; }).attr('stroke', '#2a1f2c').attr('stroke-width', 0.6).on('mouseover', function (e, d) { d3.select(this).attr('stroke', '#ef4444').attr('stroke-width', 1.5); const name = fipsToStateName[String(d.id).padStart(2, '0')]; if (basicData.por_estado[name]) { const info = basicData.por_estado[name]; tooltip.html(`<strong>❤️ ${name}</strong><br>Prevalencia: ${info.prevalencia_porcentaje}%<br>Con enfermedad: ${info.con_enfermedad.toLocaleString()}<br>Total: ${info.total_registros.toLocaleString()}`).style('display', 'block').style('left', (e.pageX + 12) + 'px').style('top', (e.pageY - 28) + 'px'); } }).on('mousemove', function (e) { tooltip.style('left', (e.pageX + 12) + 'px').style('top', (e.pageY - 28) + 'px'); }).on('mouseout', function () { d3.select(this).attr('stroke', '#2a1f2c').attr('stroke-width', 0.6); tooltip.style('display', 'none'); });
    } catch (e) { console.error('Error mapa', e); }
}


// ==================== WAFFLE CHAR Y GRÁFICO PICTOGRÁFICO  ====================
function updateBasicAnalysisSection() {
    if (!basicData) return;

    //wAFLE
    const demo = basicData.demografica;
    drawWaffleChart('chart-sexo', demo.por_sexo.Female?.total || 0, demo.por_sexo.Male?.total || 0);

    //PICTOGRAMA
    const hombre = basicData.relaciones.por_sexo.Male;
    const mujer = basicData.relaciones.por_sexo.Female;
    drawPictogramDual('chart-coronaria-sexo-analisis',
        { prevalencia_porcentaje: mujer.prevalencia_porcentaje, total: mujer.total, con_enfermedad: mujer.con_enfermedad },
        { prevalencia_porcentaje: hombre.prevalencia_porcentaje, total: hombre.total, con_enfermedad: hombre.con_enfermedad });


}
//WAFFLE CHART
function drawWaffleChart(containerId, mujeres, hombres) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    const total = mujeres + hombres;
    if (total === 0) { container.innerHTML = '<div style="color:#ef4444; text-align:center; padding:2rem;">No hay datos de sexo disponibles</div>'; return; }
    const pctMujeres = (mujeres / total) * 100;
    const pctHombres = (hombres / total) * 100;
    let mujeresCeldas = Math.round(pctMujeres);
    let hombresCeldas = Math.round(pctHombres);
    if (mujeresCeldas + hombresCeldas !== 100) mujeresCeldas = 100 - hombresCeldas;
    const size = 30, cols = 10, rows = 10;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", cols * size); svg.setAttribute("height", rows * size); svg.setAttribute("viewBox", `0 0 ${cols * size} ${rows * size}`);
    svg.classList.add("waffle-svg");
    let cell = 0;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", c * size); rect.setAttribute("y", r * size); rect.setAttribute("width", size - 1); rect.setAttribute("height", size - 1);
        rect.setAttribute("rx", 3); rect.setAttribute("fill", cell < mujeresCeldas ? "#F472B6" : "#3B82F6");
        svg.appendChild(rect); cell++;
    }
    container.appendChild(svg);
    const labelDiv = document.createElement("div");
    labelDiv.style.textAlign = "center"; labelDiv.style.marginTop = "12px"; labelDiv.style.fontSize = "0.9rem";
    labelDiv.innerHTML = `<span style="color:#F472B6;">🌸 Mujer ${pctMujeres.toFixed(1)}%</span> &nbsp;|&nbsp; <span style="color:#3B82F6;">💙 Hombre ${pctHombres.toFixed(1)}%</span>`;
    container.appendChild(labelDiv);
}

// GRÁFICO PICTOGRÁFICO DUAL
function drawPictogramDual(containerId, mujerData, hombreData) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'pictogram-dual';

    // Tarjeta Mujer
    const mujerCard = document.createElement('div');
    mujerCard.className = 'pictogram-card';
    mujerCard.innerHTML = `
                <div class="gender-icon female"><i class="fas fa-venus"></i></div>
                <div class="gender-label">Mujer</div>
                <div class="bar-container" id="bar-container-female">
                    <div class="bar-fill" id="bar-fill-female" style="width: 0%;"></div>
                    <div class="marker" id="marker-female" style="left: 0%;"></div>
                </div>
                <div class="prevalence-text" id="prevalence-female">${mujerData.prevalencia_porcentaje.toFixed(2)}%</div>
                <div class="case-count">${mujerData.con_enfermedad.toLocaleString()} / ${mujerData.total.toLocaleString()} casos</div>
            `;
    // Tarjeta Hombre
    const hombreCard = document.createElement('div');
    hombreCard.className = 'pictogram-card';
    hombreCard.innerHTML = `
                <div class="gender-icon male"><i class="fas fa-mars"></i></div>
                <div class="gender-label">Hombre</div>
                <div class="bar-container" id="bar-container-male">
                    <div class="bar-fill" id="bar-fill-male" style="width: 0%;"></div>
                    <div class="marker" id="marker-male" style="left: 0%;"></div>
                </div>
                <div class="prevalence-text" id="prevalence-male">${hombreData.prevalencia_porcentaje.toFixed(2)}%</div>
                <div class="case-count">${hombreData.con_enfermedad.toLocaleString()} / ${hombreData.total.toLocaleString()} casos</div>
            `;
    wrapper.appendChild(mujerCard);
    wrapper.appendChild(hombreCard);
    container.appendChild(wrapper);

    // Anima las barras y marcadores después de insertar en el DOM
    setTimeout(() => {
        const barFillFemale = document.getElementById('bar-fill-female');
        const markerFemale = document.getElementById('marker-female');
        const barFillMale = document.getElementById('bar-fill-male');
        const markerMale = document.getElementById('marker-male');
        if (barFillFemale) barFillFemale.style.width = `${mujerData.prevalencia_porcentaje}%`;
        if (markerFemale) markerFemale.style.left = `${mujerData.prevalencia_porcentaje}%`;
        if (barFillMale) barFillMale.style.width = `${hombreData.prevalencia_porcentaje}%`;
        if (markerMale) markerMale.style.left = `${hombreData.prevalencia_porcentaje}%`;
    }, 50);
}




// ==================== FACTORES DE RIESGO  ====================
function riskFactor() {
    // Timeline
    const timelineElem = document.getElementById('chart-timeline-age');
    if (timelineElem) {
        const trace = {
            x: riskData.timeline.map(d => d.age_label),
            y: riskData.timeline.map(d => d.risk),
            type: 'scatter',
            mode: 'lines+markers',
            marker: { color: '#ef4444', size: 10 },
            line: { color: '#f97316', width: 3 },
            fill: 'tozeroy',
            fillcolor: 'rgba(239, 68, 68, 0.15)'
        };
        const layout = {
            autosize: true,
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#cbd5e1' },
            xaxis: { title: 'Edad', gridcolor: '#2d2a3a' },
            yaxis: { title: 'Prevalencia (%)', gridcolor: '#2d2a3a' }
        };
        Plotly.newPlot('chart-timeline-age', [trace], layout, { responsive: true });
    }

    // Headmap
    const factorGroups = {
        'Enfermedades': [
            "Prediabetes", "Diabetes",
            "ACV",
            "EPOC",
            "Enfermedad Renal",
            "Artritis",
            "Asma",
            "Cáncer de Piel"
        ],
        'Conductuales': [
            "Fuma",
            "Sedentario",
            "Obesidad",
            "No Duerme Bien",
            "Depresión"
        ]
    };
    const select = document.getElementById('factorGroupSelect');
    if (select) {
        // Dibujar el grupo por defecto (Enfermedades)
        drawHeatmap(factorGroups[select.value]);
        // Cambiar al seleccionar otra opción
        select.addEventListener('change', function () {
            drawHeatmap(factorGroups[this.value]);
        });
    }

    const comorbElem = document.getElementById('chart-comorb');
    if (comorbElem && riskData.comorbidity_matrix) {
        console.log(riskData.comorbidity_matrix);
        const data = riskData.comorbidity_matrix;
        const pairs = Object.keys(data);

        // Extraer valores
        const x = pairs;                         // etiquetas de las comorbilidades
        const y = pairs.map(p => data[p].risk);  // riesgo (%)
        const population = pairs.map(p => data[p].population);
        const cases = pairs.map(p => data[p].cases);

        // Escalar tamaños 
        const maxPop = Math.max(...population);
        const minPop = Math.min(...population);
        const sizes = population.map(p => 20 + (p - minPop) / (maxPop - minPop) * 80); // entre 20 y 100

        const trace = {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: sizes,
                color: y,
                colorscale: 'YlOrRd_r',   // amarillo (bajo) a rojo (alto)
                showscale: true,
                colorbar: { title: 'Riesgo (%)' },
                sizemode: 'area',
                sizeref: 2. * Math.max(...sizes) / (100 ** 2),  
                line: { width: 1, color: '#2d2a3a' }
            },
            text: pairs.map(p => `${p}<br>Población: ${population[pairs.indexOf(p)].toLocaleString()}<br>Casos: ${cases[pairs.indexOf(p)].toLocaleString()}`),
            hoverinfo: 'text',
            hovertemplate: '%{text}<extra></extra>'
        };

        const layout = {
            xaxis: {

                tickangle: -45,
                automargin: true,
                gridcolor: '#2d2a3a',
                tickfont: { color: '#cbd5e1' }
            },
            yaxis: {
                title: 'Prevalencia de enfermedad coronaria (%)',
                gridcolor: '#2d2a3a',
                tickfont: { color: '#cbd5e1' }
            },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#cbd5e1', family: 'Arial, sans-serif' },
            width: 1200,
            height: 600,
            margin: { l: 80, r: 30, b: 150, t: 50 }
        };

        Plotly.newPlot('chart-comorb', [trace], layout, { responsive: true });
    }
}

function drawHeatmap(selectedFactors) {
    const heatmapElem = document.getElementById('chart-headmap');
    if (!heatmapElem || !riskData.heatmap) return;

    // 1. Obtener todas las edades únicas y ordenarlas
    const ages = [...new Map(riskData.heatmap.map(item => [item.age, item.age])).keys()];
    const ageOrder = [
        "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
    ];
    const sortedAges = ageOrder.filter(age => ages.includes(age));

    // 2. Filtrar los factores por los seleccionados
    const uniqueFactors = [...new Map(riskData.heatmap.map(item => [item.factor, item.factor])).keys()];
    const sortedFactors = selectedFactors.filter(f => uniqueFactors.includes(f));

    // 3. Construir la matriz de riesgos
    const zMatrix = sortedFactors.map(factor => {
        return sortedAges.map(age => {
            const entry = riskData.heatmap.find(d => d.age === age && d.factor === factor);
            return entry ? entry.risk : null;
        });
    });

    // 4. Configuración del trace
    const trace = {
        zmin: 0,     
        zmax: 25,
        x: sortedAges,
        y: sortedFactors,
        z: zMatrix,
        type: 'heatmap',
        colorscale: 'YlOrRd_r',
        showscale: true,
        text: zMatrix.map(row => row.map(v => v !== null ? v.toFixed(1) + '%' : '')),
        texttemplate: '%{text}',
        textfont: { size: 9, color: 'white' },
        hovertemplate: 'Edad: %{x}<br>Factor: %{y}<br>Prevalencia: %{z:.1f}%<extra></extra>'
    };

    // Altura dinámica según número de filas
    const rowCount = sortedFactors.length;

    // 5. Layout con tema oscuro/transparente
    const layout = {
        autosize: true,
        width: 1100,
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', family: 'Arial, sans-serif' },
        xaxis: {
            title: { text: 'Grupo de edad', font: { color: '#cbd5e1' } },
            tickangle: -45,
            gridcolor: '#2d2a3a',
            tickfont: { color: '#cbd5e1' },
            linecolor: '#2d2a3a',
            zerolinecolor: '#2d2a3a'
        },
        yaxis: {
            title: { font: { color: '#cbd5e1' } },
            gridcolor: '#2d2a3a',
            tickfont: { color: '#cbd5e1' },
            automargin: true,
            linecolor: '#2d2a3a'
        },
        coloraxis: {
            colorbar: {
                title: { text: 'Prevalencia (%)', font: { color: '#cbd5e1' } },
                tickfont: { color: '#cbd5e1' },
                ticks: 'outside'
            }
        }
    };

    // 6. Renderizar
    Plotly.newPlot('chart-headmap', [trace], layout, { responsive: true });
}


// ==================== FACTORES DE RIESGO  ====================
function mlAnalisis() {
    // ---------- RANDOM FOREST ----------
    const rf = mlRFData;
    const rfMetrics = rf.metricas_modelo;
    const rfCM = rf.matriz_confusion;
    const rfPred = rf.predicciones_dataset;
    const rfFeatures = rf.caracteristicas_importantes
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10);

    // Métricas RF
    const rfMetricsHtml = `
        <div class="metrics-row">
            <div class="stat-card-sm"><div class="stat-value-sm">${(rfMetrics.accuracy * 100).toFixed(1)}%</div><div class="stat-label-sm">Accuracy</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(rfMetrics.recall * 100).toFixed(1)}%</div><div class="stat-label-sm">Recall (Sensibilidad)</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(rfMetrics.precision * 100).toFixed(1)}%</div><div class="stat-label-sm">Precision</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(rfMetrics.f1_score * 100).toFixed(1)}%</div><div class="stat-label-sm">F1-Score</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(rfMetrics.auc_roc * 100).toFixed(1)}%</div><div class="stat-label-sm">AUC-ROC</div></div>
        </div>
    `;
    document.getElementById('rf-metrics').innerHTML = rfMetricsHtml;

    // Matriz de confusión
    const rfRightHtml = `
        <div class="column-card">
            <h3>📊 Matriz de Confusión</h3>
            <div class="confusion-matrix-visual">
                <div class="confusion-cell true-negative">
                    <div class="cell-label">Verdaderos Negativos</div>
                    <div class="cell-value">${rfCM.verdaderos_negativos.toLocaleString()}</div>
                </div>
                <div class="confusion-cell false-positive">
                    <div class="cell-label">Falsos Positivos</div>
                    <div class="cell-value">${rfCM.falsos_positivos.toLocaleString()}</div>
                </div>
                <div class="confusion-cell false-negative">
                    <div class="cell-label">Falsos Negativos</div>
                    <div class="cell-value">${rfCM.falsos_negativos.toLocaleString()}</div>
     
                </div>
                <div class="confusion-cell true-positive">
                    <div class="cell-label">Verdaderos Positivos</div>
                    <div class="cell-value">${rfCM.verdaderos_positivos.toLocaleString()}</div>
                </div>
            </div>
            <div class="confusion-note">🔍 ${rfCM.interpretacion.falsos_negativos}</div>
            <div class="predictions-box">
                <h3>📈 Predicciones sobre el dataset completo</h3>
                <p><strong>${rfPred.predichos_con_enfermedad.toLocaleString()}</strong> personas (${rfPred.porcentaje_predicho_con_enfermedad}%) fueron clasificadas con riesgo de enfermedad coronaria.</p>
                <p>${rfPred.nota_importante}</p>
            </div>
        </div>
    `;
    document.getElementById('rf-right').innerHTML = rfRightHtml;

    // Gráfico de barras horizontales RF - CON AJUSTES DE HORIZONTALIDAD
    const rfHorizontalTrace = {
        y: rfFeatures.map(f => f.feature),
        x: rfFeatures.map(f => f.importance),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#f97316' },
        text: rfFeatures.map(f => f.importance.toFixed(4)),
        textposition: 'outside',
        textfont: { size: 9 },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    };
    const rfHorizLayout = {
        title: { text: '🏆 Top 10 Características (Random Forest)', font: { size: 13, color: '#cbd5e1' } },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', size: 10 },
        xaxis: { gridcolor: '#2d2a3a', zerolinecolor: '#2d2a3a', tickfont: { size: 9 } },
        yaxis: {
            automargin: true,
            tickfont: { size: 9 },
            gridcolor: '#2d2a3a',
            categoryorder: 'total ascending'
        },
        margin: { l: 180, r: 60, t: 40, b: 30 },
        height: 450,
        width: null,  // se adapta al contenedor
        autosize: true
    };
    Plotly.newPlot('rf-horizontal-chart', [rfHorizontalTrace], rfHorizLayout, { responsive: true });

    // ---------- XGBOOST ----------
    const xgb = mlXBData;
    const xgbMetrics = xgb.capacidad_predictiva;
    const xgbCM = xgb.matriz_confusion;
    const xgbPred = xgb.predicciones_dataset;
    const xgbFeatures = xgb.caracteristicas_importantes
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10);

    // Métricas XGB
    const xgbMetricsHtml = `
        <div class="metrics-row">
            <div class="stat-card-sm"><div class="stat-value-sm">${(xgbMetrics.accuracy.valor * 100).toFixed(1)}%</div><div class="stat-label-sm">Accuracy</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(xgbMetrics.recall.valor * 100).toFixed(1)}%</div><div class="stat-label-sm">Recall</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(xgbMetrics.precision.valor * 100).toFixed(1)}%</div><div class="stat-label-sm">Precision</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(2 * xgbMetrics.precision.valor * xgbMetrics.recall.valor / (xgbMetrics.precision.valor + xgbMetrics.recall.valor) * 100).toFixed(1)}%</div><div class="stat-label-sm">F1-Score</div></div>
            <div class="stat-card-sm"><div class="stat-value-sm">${(xgbMetrics.auc_roc.valor * 100).toFixed(1)}%</div><div class="stat-label-sm">AUC-ROC</div></div>
        </div>
    `;
    document.getElementById('xgb-metrics').innerHTML = xgbMetricsHtml;

    // Matriz de confusión XGB
    const xgbRightHtml = `
        <div class="column-card">
            <h3>📊 Matriz de Confusión</h3>
            <div class="confusion-matrix-visual">
                <div class="confusion-cell true-negative">
                    <div class="cell-label">Verdaderos Negativos</div>
                    <div class="cell-value">${xgbCM.verdaderos_negativos.cantidad.toLocaleString()}</div>
                </div>
                <div class="confusion-cell false-positive">
                    <div class="cell-label">Falsos Positivos</div>
                    <div class="cell-value">${xgbCM.falsos_positivos.cantidad.toLocaleString()}</div>
                </div>
                <div class="confusion-cell false-negative">
                    <div class="cell-label">Falsos Negativos</div>
                    <div class="cell-value">${xgbCM.falsos_negativos.cantidad.toLocaleString()}</div>
                </div>
                <div class="confusion-cell true-positive">
                    <div class="cell-label">Verdaderos Positivos</div>
                    <div class="cell-value">${xgbCM.verdaderos_positivos.cantidad.toLocaleString()}</div>
                </div>
            </div>
            <div class="confusion-note">⚠️ ${xgbCM.falsos_negativos.riesgo}</div>
            <div class="predictions-box">
                <h3>📈 Predicciones sobre el dataset</h3>
                <p> <strong>${xgbPred.predichos_con_enfermedad.toLocaleString()}</strong> personas (${xgbPred.porcentaje_predicho}%) con riesgo alto.</p>
                <p>Hallazgo clave: <strong>${xgbPred.hallazgos_clave.alto_riesgo_sin_diagnostico.cantidad.toLocaleString()}</strong> personas sin diagnóstico previo tienen >50% de probabilidad.</p>
            </div>
        </div>
    `;
    document.getElementById('xgb-right').innerHTML = xgbRightHtml;

    // Gráfico de barras horizontales XGB 
    const xgbHorizontalTrace = {
        y: xgbFeatures.map(f => f.feature),
        x: xgbFeatures.map(f => f.importance),
        type: 'bar',
        orientation: 'h',
        marker: { color: '#3b82f6' },
        text: xgbFeatures.map(f => f.importance.toFixed(4)),
        textposition: 'outside',
        textfont: { size: 9 },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    };
    const xgbHorizLayout = {
        title: { text: '🏆 Top 10 Características (XGBoost)', font: { size: 13, color: '#cbd5e1' } },
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', size: 10 },
        xaxis: { gridcolor: '#2d2a3a', zerolinecolor: '#2d2a3a', tickfont: { size: 9 } },
        yaxis: {
            automargin: true,
            tickfont: { size: 9 },
            gridcolor: '#2d2a3a',
            categoryorder: 'total ascending'
        },
        margin: { l: 180, r: 60, t: 40, b: 30 },

        width: 550,
        autosize: true
    };
    Plotly.newPlot('xgb-horizontal-chart', [xgbHorizontalTrace], xgbHorizLayout, { responsive: true });

    // ------------- TAB 3: ANÁLISIS COMPARATIVO -------------
    const rfRecallReal = (mlRFData.metricas_modelo.recall * 100).toFixed(1);
    const xgbRecallReal = (mlXBData.capacidad_predictiva.recall.valor * 100).toFixed(1);

    let rfRecall, xgbRecall, mejoraPorcentaje;

    rfRecall = parseFloat(rfRecallReal);
    xgbRecall = parseFloat(xgbRecallReal);
    mejoraPorcentaje = ((xgbRecall - rfRecall) / rfRecall * 100).toFixed(1);


    const rfAcc = (mlRFData.metricas_modelo.accuracy * 100).toFixed(1);
    const rfPrec = (mlRFData.metricas_modelo.precision * 100).toFixed(1);
    const rfAuc = (mlRFData.metricas_modelo.auc_roc * 100).toFixed(1);
    const xgbAcc = (mlXBData.capacidad_predictiva.accuracy.valor * 100).toFixed(1);
    const xgbPrec = (mlXBData.capacidad_predictiva.precision.valor * 100).toFixed(1);
    const xgbAuc = (mlXBData.capacidad_predictiva.auc_roc.valor * 100).toFixed(1);

    const riesgoOcultoRF = mlRFData.predicciones_dataset.predichos_con_enfermedad.toLocaleString();
    const riesgoOcultoXGB = mlXBData.predicciones_dataset.hallazgos_clave.alto_riesgo_sin_diagnostico.cantidad.toLocaleString();

    const comparativeHtml = `
    <div class="analysis-grid">
        <div class="analysis-card">
            <div class="card-header">🌲 Random Forest</div>
            <div class="metric-list">
                <div class="metric-item"><span class="metric-label">Accuracy:</span><span class="metric-value">${rfAcc}%</span></div>
                <div class="metric-item"><span class="metric-label">Precision:</span><span class="metric-value">${rfPrec}%</span></div>
                <div class="metric-item"><span class="metric-label">Recall:</span><span class="metric-value">${rfRecall}%</span></div>
                <div class="metric-item"><span class="metric-label">AUC-ROC:</span><span class="metric-value">${rfAuc}%</span></div>
            </div>
            <div style="margin-top: 12px; font-size:0.85rem;">
                📈 Detecta <strong>${rfRecall}</strong> de cada 100 enfermos.<br>
                🏅 <strong>${riesgoOcultoRF}</strong> personas en riesgo oculto.
            </div>
        </div>

        <div class="analysis-card">
            <div class="card-header">⚡ XGBoost <span style="font-size:0.7rem;">(Mejorado)</span></div>
            <div class="metric-list">
                <div class="metric-item"><span class="metric-label">Accuracy:</span><span class="metric-value">${xgbAcc}%</span></div>
                <div class="metric-item"><span class="metric-label">Precision:</span><span class="metric-value">${xgbPrec}%</span></div>
                <div class="metric-item"><span class="metric-label">Recall:</span><span class="metric-value">${xgbRecall}%</span></div>
                <div class="metric-item"><span class="metric-label">AUC-ROC:</span><span class="metric-value">${xgbAuc}%</span></div>
            </div>
            <div style="margin-top: 12px; font-size:0.85rem;">
                🔄 <span class="highlight">+${mejoraPorcentaje}% mejora en Recall</span><br>
                📈 Detecta <strong>${xgbRecall}</strong> de cada 100 enfermos.<br>
                🔄 <strong>${riesgoOcultoXGB}</strong> personas identificadas.
            </div>
        </div>

        <div class="analysis-card">
            <div class="card-header">📊 Comparación: Recall</div>
            <div id="recall-compare-chart" class="recall-chart" style="width:100%; height:200px;"></div>
            <p style="font-size:0.8rem; text-align:center; margin-top:8px;">
                XGBoost supera a Random Forest en detección de casos reales.
            </p>
        </div>

        <div class="analysis-card">
            <div class="card-header">🏥 Conclusiones prácticas</div>
            <ul class="conclusion-list">
                <li>✅ XGBoost identifica <strong>${riesgoOcultoXGB}</strong> personas con alta probabilidad sin diagnóstico.</li>
                <li> Precisión baja (${xgbPrec}%) pero alto recall (${xgbRecall}%) ideal para tamizaje poblacional.</li>
                <li> Recomendación: priorizar pruebas diagnósticas (ECG, ecocardiograma) en esas ${riesgoOcultoXGB} personas.</li>
                <li> El modelo <strong>no es diagnóstico definitivo</strong>, complementar con evaluación clínica.</li>
            </ul>
        </div>
    </div>
    `;
    document.getElementById('comparative-content').innerHTML = comparativeHtml;

    // Gráfico de comparación de recall 
    const recallCompareTrace = {
        x: ['Random Forest', 'XGBoost'],
        y: [rfRecall, xgbRecall],
        type: 'bar',
        marker: { color: ['#f97316', '#3b82f6'] },
        text: [`${rfRecall}%`, `${xgbRecall}%`],
        textposition: 'auto',
        textfont: { color: 'white', size: 11 }
    };
    const recallCompareLayout = {
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#cbd5e1', size: 10 },
        yaxis: { title: 'Recall (%)', range: [0, 100], gridcolor: '#2d2a3a', showgrid: true, zeroline: false },
        xaxis: { title: '', gridcolor: '#2d2a3a', tickfont: { size: 10 } },
        margin: { l: 35, r: 20, t: 20, b: 25 },
        autosize: true,
        width: 350
    };
    Plotly.newPlot('recall-compare-chart', [recallCompareTrace], recallCompareLayout, { responsive: true });
}