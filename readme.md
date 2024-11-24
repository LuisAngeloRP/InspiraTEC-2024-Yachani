
# 📚 Yachani - Biblioteca Educativa

**Yachani** es una plataforma educativa diseñada para hacer que los recursos educativos de calidad sean accesibles para todos, utilizando tecnología avanzada de procesamiento de lenguaje natural.

---

## 🎯 Nuestra Misión

Hacer que el conocimiento sea **accesible**, **organizado** y **fácil de encontrar** para estudiantes de todas partes, sin importar su ubicación o recursos disponibles.

---

## 📊 Estadísticas de la Biblioteca

Nuestra biblioteca ofrece una amplia gama de documentos educativos **categorizados** y **organizados**, facilitando su acceso y uso para todos.

---

## 🚀 Cómo Empezar

### 📋 Prerrequisitos

- **Python** 3.12 o superior.
- **pip**, el gestor de paquetes de Python.
- Una **clave de API de OpenAI** para el procesamiento de lenguaje natural.

### ⚙️ Instalación

1. **Clona el repositorio:**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

2. **Crea y activa un entorno virtual:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Windows usa `.venv\Scripts\activate`
    ```

3. **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configura las variables de entorno:**

    - Copia el archivo de ejemplo `.env.example` y ajústalo a tus configuraciones en un nuevo archivo `.env`:

      ```bash
      cp .env.example .env
      ```

    - En el archivo `.env`, agrega tu clave de API de OpenAI:

      ```env
      OPENAI_API_KEY=tu_clave_de_openai
      ```

### ▶️ Uso

Para iniciar la aplicación, ejecuta:

```bash
streamlit run Home.py
```

Esto abrirá la aplicación en tu navegador predeterminado.

---

## 📂 Estructura del Proyecto

- **`Home.py`**: Página principal de la aplicación.
- **`pages/`**: Contiene las diferentes páginas de la aplicación.
- **`data/`**: Almacena los datos y metadatos de los documentos.
- **`utils/`**: Funciones auxiliares y utilidades.
- **`Yachani_app/`**: Configuración principal de la aplicación.


---

## 🌟 Créditos

Desarrollado con ❤️ por el equipo de **Yachani**.
