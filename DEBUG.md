# Отладка проблем с авторизацией

## Проблема: После нажатия на любую кнопку заставляет снова вводить пароль

### Шаги для диагностики:

1. **Откройте консоль браузера** (F12 → Console)
2. **Откройте терминал с backend** (там где запущен Python/Flask)
3. **Войдите в систему**
4. **Попробуйте выполнить действие** (например, загрузить видео)
5. **Посмотрите логи**

### Что искать в логах:

#### В консоли браузера (F12):
```
API Error: {
  status: 401 или 422,
  url: "/api/videos/upload",
  method: "post",
  errorData: { error: "...", msg: "..." }
}
```

**Если видите:**
- `status: 401` + `msg: "Token has expired"` → Токен истёк (нужно перелогиниться)
- `status: 422` + `msg: "Signature verification failed"` → Токен невалидный
- `status: 401` + `msg: "Authorization header is missing"` → Токен не отправляется

#### В терминале backend:
```
INFO:werkzeug: POST /api/videos/upload
INFO:__main__: Video upload request from user: 123
INFO:__main__: Content-Type: multipart/form-data; boundary=...
INFO:__main__: Files: ['file']
INFO:__main__: Role check for user ID: 123, required role: ['teacher', 'admin']
INFO:__main__: Role check passed for user 123 (username)
```

**Если видите:**
- `WARNING: Expired token` → Токен истёк
- `WARNING: Invalid token` → Токен битый
- `WARNING: Missing token` → Токен не пришёл на сервер
- `ERROR: Error in role_required decorator` → Ошибка проверки прав

### Частые причины и решения:

#### 1. Токен не сохраняется после логина
**Проверка:**
```javascript
// В консоли браузера
localStorage.getItem('token')
localStorage.getItem('user')
```

**Решение:** 
- Очистите localStorage: `localStorage.clear()`
- Перелогиньтесь

#### 2. Токен истекает слишком быстро
**Проверка:** В `backend/config.py` проверьте:
```python
JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)  # Должно быть 24 часа
```

**Решение:** Увеличьте время жизни токена

#### 3. Токен не отправляется в запросах
**Проверка:** В консоли браузера → Network → выберите запрос → Headers
Должен быть заголовок:
```
Authorization: Bearer <токен>
```

**Решение:** Проверьте `frontend/src/services/api.js` → request interceptor

#### 4. Backend не валидирует токен правильно
**Проверка:** Посмотрите в логи backend на `WARNING` сообщения

**Решение:** 
- Убедитесь, что `JWT_SECRET_KEY` одинаковый
- Перезапустите backend
- Перелогиньтесь

#### 5. CORS проблемы
**Проверка:** В консоли браузера ищите ошибки CORS

**Решение:** В `backend/app.py` проверьте:
```python
CORS(app)  # Должно быть включено
```

### Быстрое решение:

1. **Остановите оба сервера** (Ctrl+C)
2. **Очистите localStorage:**
   - Откройте http://localhost:3000
   - F12 → Console
   - Введите: `localStorage.clear()`
   - Обновите страницу (F5)
3. **Перезапустите серверы:**
   ```bash
   # В корне проекта
   start.bat  # (Windows)
   # или
   ./start.sh  # (Linux/Mac)
   ```
4. **Залогиньтесь заново**
5. **Проверьте в консоли:**
   ```javascript
   localStorage.getItem('token')  // Должен быть токен
   localStorage.getItem('user')   // Должен быть JSON с данными пользователя
   ```

### Если проблема сохраняется:

Отправьте разработчику:
1. Скриншот консоли браузера с ошибкой
2. Логи из терминала backend
3. Версию Python (`python --version`)
4. Версию Node.js (`node --version`)
